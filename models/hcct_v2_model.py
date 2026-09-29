import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import os

class HCCTModel_v2:
    def __init__(self, n_cards=32):
        self.n_cards = n_cards
        
    def _aggregate_to_trials(self, df):
        """Схлопывает пошаговые логи в уровень триалов для соответствия формуле Softmax."""
        # Группируем логи по триалам
        df_trials = df.groupby(['user_id', 'trial_number'], as_index=False).agg({
            'flips': 'max',          # Фактическое итоговое количество перевернутых карт (K)
            'popped': 'max',         # Был ли штраф в этом триале (True/False конвертируется в 1/0 позже)
            'gain_amount': 'first',
            'loss_amount': 'first',
            'loss_cards': 'first'
        })
        # В пошаговой версии hCCT итоговый выбор N совпадает с фактическим K (где участник остановился или взорвался)
        df_trials['choice'] = df_trials['flips']
        return df_trials.sort_values(['user_id', 'trial_number'])

    def prep_data(self, df):
        """Подготовка 2D матриц для быстрого scan (trials x subjects)"""
        # 1. Агрегируем данные
        df_trials = self._aggregate_to_trials(df)
        
        users = df_trials['user_id'].unique()
        n_subj = len(users)
        max_trials = df_trials.groupby('user_id').size().max()
        
        choices = np.zeros((max_trials, n_subj), dtype=np.int32)
        Ks = np.zeros((max_trials, n_subj), dtype=np.int32)
        is_loss = np.zeros((max_trials, n_subj), dtype=np.int32)
        gains = np.zeros((max_trials, n_subj), dtype=np.float64)
        losses = np.zeros((max_trials, n_subj), dtype=np.float64)
        masks = np.zeros((max_trials, n_subj), dtype=np.float64)
        P_obj = np.zeros((max_trials, n_subj, self.n_cards), dtype=np.float64)

        for i, uid in enumerate(users):
            udf = df_trials[df_trials['user_id'] == uid]
            t_len = len(udf)
            
            choices[:t_len, i] = udf['choice'].values 
            Ks[:t_len, i] = udf['flips'].values
            is_loss[:t_len, i] = udf['popped'].astype(int).values
            gains[:t_len, i] = udf['gain_amount'].values
            losses[:t_len, i] = udf['loss_amount'].values
            masks[:t_len, i] = 1.0
            
            loss_cards = udf['loss_cards'].values
            for t in range(t_len):
                p_n = np.clip(1.0 - np.arange(1, self.n_cards + 1) * loss_cards[t] / self.n_cards, 1e-6, 1.0)
                P_obj[t, i, :] = p_n
                
            for t in range(t_len, max_trials):
                P_obj[t, i, :] = P_obj[t_len-1, i, :]

        return choices, Ks, is_loss, gains, losses, masks, P_obj, users

    
    def build_model(self, choices, Ks, is_losses, gains, losses, masks, P_objs):
        max_trials, n_subj = choices.shape
        
        with pm.Model() as model:
            # 1. Групповой уровень гиперпараметров
            mu_rho = pm.Normal('mu_rho', 0, 1.5)
            sigma_rho = pm.HalfNormal('sigma_rho', 1.0)
            mu_lam = pm.Normal('mu_lam', 0, 1.5)
            sigma_lam = pm.HalfNormal('sigma_lam', 1.0)
            mu_theta = pm.Normal('mu_theta', 0, 1.5)
            sigma_theta = pm.HalfNormal('sigma_theta', 1.0)
            mu_Arew = pm.Normal('mu_Arew', 0, 1.5)
            sigma_Arew = pm.HalfNormal('sigma_Arew', 1.0)
            mu_Apun = pm.Normal('mu_Apun', 0, 1.5)
            sigma_Apun = pm.HalfNormal('sigma_Apun', 1.0)
            mu_gamma = pm.Normal('mu_gamma', 0, 1.5)
            sigma_gamma = pm.HalfNormal('sigma_gamma', 1.0)

            # 2. Индивидуальный уровень (Non-centered)
            rho_raw = pm.Normal('rho_raw', 0, 1, shape=n_subj)
            lam_raw = pm.Normal('lam_raw', 0, 1, shape=n_subj)
            theta_raw = pm.Normal('theta_raw', 0, 1, shape=n_subj)
            Arew_raw = pm.Normal('Arew_raw', 0, 1, shape=n_subj)
            Apun_raw = pm.Normal('Apun_raw', 0, 1, shape=n_subj)
            gamma_raw = pm.Normal('gamma_raw', 0, 1, shape=n_subj)

            # 3. Трансформации в классические границы
            rho = pm.Deterministic('rho', 3.0 * pm.math.invlogit(mu_rho + sigma_rho * rho_raw))
            lam = pm.Deterministic('lam', 10.0 * pm.math.invlogit(mu_lam + sigma_lam * lam_raw))
            theta = pm.Deterministic('theta', 10.0 * pm.math.invlogit(mu_theta + sigma_theta * theta_raw))
            Arew = pm.Deterministic('Arew', pm.math.invlogit(mu_Arew + sigma_Arew * Arew_raw))
            Apun = pm.Deterministic('Apun', pm.math.invlogit(mu_Apun + sigma_Apun * Apun_raw))
            gamma = pm.Deterministic('gamma', 5.0 * pm.math.invlogit(mu_gamma + sigma_gamma * gamma_raw))

            # Перевод данных в тензоры
            choices_t = pt.as_tensor_variable(choices)
            Ks_t = pt.as_tensor_variable(Ks)
            is_losses_t = pt.as_tensor_variable(is_losses)
            gains_t = pt.as_tensor_variable(gains)
            losses_t = pt.as_tensor_variable(losses)
            masks_t = pt.as_tensor_variable(masks)
            P_objs_t = pt.as_tensor_variable(P_objs)

            # Вектор N (1, 2... 32)
            N_vec = pt.arange(1, self.n_cards + 1, dtype='float64')

            def step(choice, K, is_loss, gain, loss, mask, P_obj, 
                     P_prev, 
                     rho, lam, Arew, Apun, gamma, theta):
                
                N_mat = N_vec.dimshuffle('x', 0) # (1, 32)
                
                # Полезность
                u_gain = pt.power(N_mat * gain.dimshuffle(0, 'x'), rho.dimshuffle(0, 'x'))
                u_loss = -lam.dimshuffle(0, 'x') * pt.power(pt.abs(loss.dimshuffle(0, 'x')), rho.dimshuffle(0, 'x'))
                
                # Ожидаемая полезность
                EU = P_prev * u_gain + (1 - P_prev) * u_loss

                scale_factor = 100.0 # или 100.0, подберите под масштаб ваших баллов
                EU_scaled = EU / scale_factor
                
                # Softmax
                theta_mat = theta.dimshuffle(0, 'x')
                EU_max = pt.max(EU_scaled, axis=1, keepdims=True)
                exp_EU = pt.exp(theta_mat * (EU_scaled - EU_max))
                probs = exp_EU / pt.sum(exp_EU, axis=1, keepdims=True)
                
                # Log-likelihood
                subj_idx = pt.arange(probs.shape[0])
                chosen_idx = pt.cast(choice - 1, 'int32')
                # prob_chosen = probs[subj_idx, chosen_idx]
                # ll = pt.log(prob_chosen + 1e-12) * mask
                probs_clipped = pt.clip(probs, 1e-6, 1.0 - 1e-6)
                prob_chosen = probs_clipped[subj_idx, chosen_idx]
                ll = pt.log(prob_chosen) * mask
                
                # Обновление P[cite: 1]
                K_idx = pt.cast(K - 1, 'int32')
                p_K = P_obj[subj_idx, K_idx].dimshuffle(0, 'x')
                dist = pt.abs(P_obj - p_K)
                G = pt.exp(-gamma.dimshuffle(0, 'x') * dist)
                
                K_mat = K.dimshuffle(0, 'x')
                N_le_K = pt.cast(N_mat <= K_mat, 'float64')
                N_gt_K = pt.cast(N_mat > K_mat, 'float64')
                N_ge_K = pt.cast(N_mat >= K_mat, 'float64')
                N_lt_K = pt.cast(N_mat < K_mat, 'float64')
                
                Arew_mat = Arew.dimshuffle(0, 'x')
                Apun_mat = Apun.dimshuffle(0, 'x')
                
                # Обновления при успехе[cite: 1]
                P_rew = N_le_K * (P_prev + Arew_mat * (1 - P_prev)) + \
                        N_gt_K * (P_prev + Arew_mat * G * (1 - P_prev))
                        
                # Обновления при проигрыше[cite: 1]
                P_pun = N_ge_K * (P_prev + Apun_mat * (0 - P_prev)) + \
                        N_lt_K * (P_prev + Arew_mat * (1 - P_prev))
                
                is_loss_mat = is_loss.dimshuffle(0, 'x')
                P_next = pt.switch(is_loss_mat > 0, P_pun, P_rew)
                
                # Паддинг защита
                mask_mat = mask.dimshuffle(0, 'x')
                P_next = pt.switch(mask_mat > 0, P_next, P_prev)
                
                return P_next, ll

            # Начальное состояние - объективная вероятность на 1 триале[cite: 1]
            P_init = P_objs_t[0]

            [P_states, LLs], _ = scan(
                fn=step,
                sequences=[choices_t, Ks_t, is_losses_t, gains_t, losses_t, masks_t, P_objs_t],
                outputs_info=[P_init, None],
                non_sequences=[rho, lam, Arew, Apun, gamma, theta]
            )

            pm.Potential('likelihood', pt.sum(LLs))
            
        return model

    def fit(self, df, draws=1000, tune=1000, chains=4, cores=4):
        c, k, i, g, l, m, p, users = self.prep_data(df)
        model = self.build_model(c, k, i, g, l, m, p)
        
        with model:
            trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, 
                              target_accept=0.95, return_inferencedata=True)
                              
        # Вывод показателя Гельмана-Рубина
        summary = az.summary(trace)
        print("\n=== Показатель Гельмана-Рубина (R-hat) ===")
        print(summary['r_hat'].describe())
        if summary['r_hat'].max() > 1.05:
            print("ВНИМАНИЕ: Некоторые цепи могли не сойтись (R-hat > 1.05)!")
            
        return trace, users

    def posterior_predictive_check(self, trace, df):
            print("\nНачинаем Posterior Predictive Check (PPC)...")
            # 1. Обязательно агрегируем данные и здесь, чтобы цикл шел по триалам, а не по кликам
            df_trials = self._aggregate_to_trials(df)
            
            rhos = trace.posterior['rho'].mean(dim=['chain', 'draw']).values
            lams = trace.posterior['lam'].mean(dim=['chain', 'draw']).values
            thetas = trace.posterior['theta'].mean(dim=['chain', 'draw']).values
            Arews = trace.posterior['Arew'].mean(dim=['chain', 'draw']).values
            Apuns = trace.posterior['Apun'].mean(dim=['chain', 'draw']).values
            gammas = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
            
            users = df_trials['user_id'].unique()
            ppc_results = []
            
            for i, uid in enumerate(users):
                udf = df_trials[df_trials['user_id'] == uid]
                rho, lam, theta = rhos[i], lams[i], thetas[i]
                Arew, Apun, gamma = Arews[i], Apuns[i], gammas[i]
                
                P_prev = np.clip(1.0 - np.arange(1, 33) * udf['loss_cards'].iloc[0] / 32, 1e-6, 1.0)
                
                y_true, y_pred = [], []
                for _, row in udf.iterrows():
                    N_vec = np.arange(1, 33)
                    u_gain = np.power(N_vec * row['gain_amount'], rho)
                    u_loss = -lam * np.power(np.abs(row['loss_amount']), rho)
                    EU = P_prev * u_gain + (1 - P_prev) * u_loss
                    
                    exp_EU = np.exp(theta * (EU - np.max(EU)))
                    probs = exp_EU / np.sum(exp_EU)
                    
                    choice = int(row['choice'])
                    pred_choice = np.argmax(probs) + 1
                    y_true.append(choice)
                    y_pred.append(pred_choice)
                    
                    K = int(row['flips'])
                    is_loss = int(row['popped'])
                    P_obj = np.clip(1.0 - np.arange(1, 33) * row['loss_cards'] / 32, 1e-6, 1.0)
                    p_K = P_obj[K-1]
                    G = np.exp(-gamma * np.abs(P_obj - p_K))
                    
                    P_next = np.zeros(32)
                    for n in range(32):
                        actual_N = n + 1
                        if is_loss == 0:
                            if actual_N <= K:
                                P_next[n] = P_prev[n] + Arew * (1 - P_prev[n])
                            else:
                                P_next[n] = P_prev[n] + Arew * G[n] * (1 - P_prev[n])
                        else:
                            if actual_N >= K:
                                P_next[n] = P_prev[n] + Apun * (0 - P_prev[n])
                            else:
                                P_next[n] = P_prev[n] + Arew * (1 - P_prev[n])
                    P_prev = P_next
                    
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                hit_rate = np.mean(y_true == y_pred)
                ppp = 1 if np.var(y_pred) > np.var(y_true) else 0
                
                ppc_results.append({'user_id': uid, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'HitRate': hit_rate, 'ppp': ppp})
                
            metrics_df = pd.DataFrame(ppc_results)
            print("\nСредние метрики PPC:")
            print(metrics_df.mean(numeric_only=True))
            return metrics_df

    
    def simulate_subject(self, params, template_df):
        """Симуляция поведения одного агента на основе истинных параметров."""
        rho, lam, theta, Arew, Apun, gamma = params
        
        sim_rows = []
        # Начальная вероятность
        P_prev = np.clip(1.0 - np.arange(1, 33) * template_df['loss_cards'].iloc[0] / 32, 1e-6, 1.0)
        
        for _, row in template_df.iterrows():
            N_vec = np.arange(1, 33)
            u_gain = np.power(N_vec * row['gain_amount'], rho)
            u_loss = -lam * np.power(np.abs(row['loss_amount']), rho)
            EU = P_prev * u_gain + (1 - P_prev) * u_loss
            
            # Softmax с защитой от переполнения
            exp_EU = np.exp(theta * (EU - np.max(EU)))
            probs = exp_EU / np.sum(exp_EU)
            
            # Агент выбирает N
            choice = np.random.choice(N_vec, p=probs)
            
            # Симуляция исхода среды
            loss_cards = row['loss_cards']
            p_loss = (choice * loss_cards) / 32.0
            p_loss = min(p_loss, 1.0)
            popped = int(np.random.rand() < p_loss)
            
            if popped:
                # Если взорвался, штраф случается на случайной карте от 1 до choice
                flips = np.random.randint(1, choice + 1)
            else:
                flips = choice
                
            sim_row = row.copy()
            sim_row['choice'] = choice
            sim_row['flips'] = flips
            sim_row['popped'] = popped
            sim_rows.append(sim_row)
            
            # Обновление вероятностей P (как в математическом ядре)
            P_obj = np.clip(1.0 - np.arange(1, 33) * loss_cards / 32, 1e-6, 1.0)
            p_K = P_obj[flips-1]
            G = np.exp(-gamma * np.abs(P_obj - p_K))
            
            P_next = np.zeros(32)
            for n in range(32):
                actual_N = n + 1
                if popped == 0:
                    if actual_N <= flips:
                        P_next[n] = P_prev[n] + Arew * (1 - P_prev[n])
                    else:
                        P_next[n] = P_prev[n] + Arew * G[n] * (1 - P_prev[n])
                else:
                    if actual_N >= flips:
                        P_next[n] = P_prev[n] + Apun * (0 - P_prev[n])
                    else:
                        P_next[n] = P_prev[n] + Arew * (1 - P_prev[n])
            P_prev = P_next
            
        return pd.DataFrame(sim_rows)

    def parameter_recovery(self, trace, df, n_subjects=50):
        """Строгий Байесовский Parameter Recovery (MCMC) с расчетом Coverage."""
        import scipy.stats as stats
        
        print(f"\n[*] Запуск Эмпирического Parameter Recovery (N={n_subjects})...")
        if trace is None:
            raise ValueError("Сначала запустите fit() на реальных данных для извлечения priors!")
            
        df_trials = self._aggregate_to_trials(df)
        unique_users = df_trials['user_id'].unique()
        
        post = trace.posterior
        chains, draws, _ = post['rho'].shape
        total_samples = chains * draws
        
        np.random.seed(42)
        sample_idx = np.random.choice(total_samples, size=n_subjects, replace=False)
        chain_idx = sample_idx // draws
        draw_idx = sample_idx % draws
        
        true_params = {'rho': [], 'lam': [], 'theta': [], 'Arew': [], 'Apun': [], 'gamma': []}
        sim_data_frames = []
        
        print(" -> Симуляция виртуальных агентов...")
        for i in range(n_subjects):
            c_idx = chain_idx[i]
            d_idx = draw_idx[i]
            rand_u = np.random.randint(0, len(unique_users)) # Случайный участник для заимствования структуры
            
            # Эмпирическое сэмплирование параметров[cite: 12]
            tp = {
                'rho': float(post['rho'][c_idx, d_idx, rand_u].values),
                'lam': float(post['lam'][c_idx, d_idx, rand_u].values),
                'theta': float(post['theta'][c_idx, d_idx, rand_u].values),
                'Arew': float(post['Arew'][c_idx, d_idx, rand_u].values),
                'Apun': float(post['Apun'][c_idx, d_idx, rand_u].values),
                'gamma': float(post['gamma'][c_idx, d_idx, rand_u].values)
            }
            
            for k in tp.keys():
                true_params[k].append(tp[k])
                
            template_df = df_trials[df_trials['user_id'] == unique_users[rand_u]].copy()
            sim_df = self.simulate_subject((tp['rho'], tp['lam'], tp['theta'], tp['Arew'], tp['Apun'], tp['gamma']), template_df)
            sim_df['user_id'] = f"sim_{i}"
            sim_data_frames.append(sim_df)
            
        sim_dataset = pd.concat(sim_data_frames, ignore_index=True)
        
        print(" -> Подгонка HBA на симулированных данных (MCMC)...")
        # Используем те же настройки NUTS
        rec_trace, _ = self.fit(sim_dataset, draws=1000, tune=1000, chains=4, cores=4)
        rec_post = rec_trace.posterior
        
        print(" -> Расчет метрик Recovery (r, R2, Bias, RMSE, Coverage)...")
        recovery_data = []
        metrics = []
        
        param_names = ['rho', 'lam', 'theta', 'Arew', 'Apun', 'gamma']
        
        for p in param_names:
            true_vals = np.array(true_params[p])
            fit_vals = rec_post[p].mean(dim=['chain', 'draw']).values
            
            # Расчет HDI для Coverage[cite: 18]
            hdi = az.hdi(rec_trace, var_names=[p])[p].values
            coverage = np.mean((true_vals >= hdi[:, 0]) & (true_vals <= hdi[:, 1]))
            
            r_val, _ = stats.pearsonr(true_vals, fit_vals)
            r2_val = r2_score(true_vals, fit_vals)
            bias = np.mean(fit_vals - true_vals)
            rmse = np.sqrt(mean_squared_error(true_vals, fit_vals))
            
            metrics.append({
                'Parameter': p,
                'r': r_val,
                'R2': r2_val,
                'Bias': bias,
                'RMSE': rmse,
                'Coverage': coverage
            })
            
            for i in range(n_subjects):
                recovery_data.append({
                    'user_id': f"sim_{i}",
                    'Parameter': p,
                    'True_Value': true_vals[i],
                    'Fit_Value': fit_vals[i],
                    'HDI_low': hdi[i, 0],
                    'HDI_high': hdi[i, 1]
                })
                
        metrics_df = pd.DataFrame(metrics)
        rec_df = pd.DataFrame(recovery_data)
        
        print("\nИтоговые метрики Parameter Recovery:")
        print(metrics_df)
        
        return rec_df, metrics_df

    # def posterior_predictive_check(self, trace, df):
    #     print("\nНачинаем Posterior Predictive Check (PPC)...")
    #     # Извлекаем средние параметры из апостериора
    #     rhos = trace.posterior['rho'].mean(dim=['chain', 'draw']).values
    #     lams = trace.posterior['lam'].mean(dim=['chain', 'draw']).values
    #     thetas = trace.posterior['theta'].mean(dim=['chain', 'draw']).values
    #     Arews = trace.posterior['Arew'].mean(dim=['chain', 'draw']).values
    #     Apuns = trace.posterior['Apun'].mean(dim=['chain', 'draw']).values
    #     gammas = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
        
    #     users = df['user_id'].unique()
    #     ppc_results = []
        
    #     for i, uid in enumerate(users):
    #         udf = df[df['user_id'] == uid].sort_values('trial_number')
    #         rho, lam, theta = rhos[i], lams[i], thetas[i]
    #         Arew, Apun, gamma = Arews[i], Apuns[i], gammas[i]
            
    #         P_prev = np.clip(1.0 - np.arange(1, 33) * udf['loss_cards'].iloc[0] / 32, 1e-6, 1.0)
            
    #         y_true, y_pred = [], []
    #         for _, row in udf.iterrows():
    #             N_vec = np.arange(1, 33)
    #             u_gain = np.power(N_vec * row['gain_amount'], rho)
    #             u_loss = -lam * np.power(np.abs(row['loss_amount']), rho)
    #             EU = P_prev * u_gain + (1 - P_prev) * u_loss
                
    #             exp_EU = np.exp(theta * (EU - np.max(EU)))
    #             probs = exp_EU / np.sum(exp_EU)
                
    #             # Сохраняем истинный выбор и предсказанный (argmax)
    #             choice = int(row['choice'])
    #             pred_choice = np.argmax(probs) + 1
    #             y_true.append(choice)
    #             y_pred.append(pred_choice)
                
    #             # Обновление
    #             K = int(row['flips'])
    #             is_loss = int(row['popped'])
    #             P_obj = np.clip(1.0 - np.arange(1, 33) * row['loss_cards'] / 32, 1e-6, 1.0)
    #             p_K = P_obj[K-1]
    #             G = np.exp(-gamma * np.abs(P_obj - p_K))
                
    #             P_next = np.zeros(32)
    #             for n in range(32):
    #                 actual_N = n + 1
    #                 if is_loss == 0:
    #                     if actual_N <= K:
    #                         P_next[n] = P_prev[n] + Arew * (1 - P_prev[n])
    #                     else:
    #                         P_next[n] = P_prev[n] + Arew * G[n] * (1 - P_prev[n])
    #                 else:
    #                     if actual_N >= K:
    #                         P_next[n] = P_prev[n] + Apun * (0 - P_prev[n])
    #                     else:
    #                         P_next[n] = P_prev[n] + Arew * (1 - P_prev[n])
    #             P_prev = P_next
                
    #         y_true = np.array(y_true)
    #         y_pred = np.array(y_pred)
            
    #         r2 = r2_score(y_true, y_pred)
    #         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    #         mae = mean_absolute_error(y_true, y_pred)
    #         hit_rate = np.mean(y_true == y_pred)
            
    #         # PPP (Bayesian p-value proxy: variance check)
    #         ppp = 1 if np.var(y_pred) > np.var(y_true) else 0
            
    #         ppc_results.append({'user_id': uid, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'HitRate': hit_rate, 'ppp': ppp})
            
    #     metrics_df = pd.DataFrame(ppc_results)
    #     print("\nСредние метрики PPC:")
    #     print(metrics_df.mean())
    #     return metrics_df