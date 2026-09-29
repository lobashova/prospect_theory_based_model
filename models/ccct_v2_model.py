import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

class CCCTModel_v2:
    def __init__(self, n_cards=33, scale_factor=10.0):
        self.n_cards = n_cards  # Опции от 0 до 32 карт
        self.scale_factor = scale_factor
        
    def fit(self, df, draws=1500, tune=1000, chains=4, cores=4):
        print(f"[*] Инициализация HBA NUTS для {df['user_id'].nunique()} участников...")
        self.users = df['user_id'].astype('category')
        self.user_codes = self.users.cat.codes.values
        self.n_subj = len(self.users.cat.categories)
        self.subj_labels = self.users.cat.categories.values
        
        choices = df['choice'].values 
        k_cards = df['k_cards_drawn'].values 
        is_penalty = df['is_penalty'].values 
        
        x_gain = df[[f'gain_n_{i}' for i in range(self.n_cards)]].values
        x_loss = df[[f'loss_n_{i}' for i in range(self.n_cards)]].values
        p_obj = df[[f'p_obj_n_{i}' for i in range(self.n_cards)]].values
        
        coords = {"subj": self.subj_labels, "opts": np.arange(self.n_cards)}
        
        with pm.Model(coords=coords) as self.model:
            # Гиперпараметры 
            mu_rho = pm.Normal('mu_rho', mu=0, sigma=1)
            sigma_rho = pm.HalfNormal('sigma_rho', sigma=1)
            mu_lam = pm.Normal('mu_lam', mu=0, sigma=1)
            sigma_lam = pm.HalfNormal('sigma_lam', sigma=1)
            mu_theta = pm.Normal('mu_theta', mu=0, sigma=1)
            sigma_theta = pm.HalfNormal('sigma_theta', sigma=1)
            mu_Arew = pm.Normal('mu_Arew', mu=0, sigma=1)
            sigma_Arew = pm.HalfNormal('sigma_Arew', sigma=1)
            mu_Apun = pm.Normal('mu_Apun', mu=0, sigma=1)
            sigma_Apun = pm.HalfNormal('sigma_Apun', sigma=1)
            mu_gamma = pm.Normal('mu_gamma', mu=0, sigma=1)
            sigma_gamma = pm.HalfNormal('sigma_gamma', sigma=1)
            
            # Z-скоры
            rho_z = pm.Normal('rho_z', mu=0, sigma=1, dims="subj")
            lam_z = pm.Normal('lam_z', mu=0, sigma=1, dims="subj")
            theta_z = pm.Normal('theta_z', mu=0, sigma=1, dims="subj")
            Arew_z = pm.Normal('Arew_z', mu=0, sigma=1, dims="subj")
            Apun_z = pm.Normal('Apun_z', mu=0, sigma=1, dims="subj")
            gamma_z = pm.Normal('gamma_z', mu=0, sigma=1, dims="subj")
            
            # Трансформации
            rho = pm.Deterministic('rho', pm.math.invlogit(mu_rho + rho_z * sigma_rho) * 3.0, dims="subj") 
            lam = pm.Deterministic('lam', pm.math.invlogit(mu_lam + lam_z * sigma_lam) * 10.0, dims="subj") 
            theta = pm.Deterministic('theta', pm.math.invlogit(mu_theta + theta_z * sigma_theta) * 10.0, dims="subj") 
            Arew = pm.Deterministic('Arew', pm.math.invlogit(mu_Arew + Arew_z * sigma_Arew), dims="subj") 
            Apun = pm.Deterministic('Apun', pm.math.invlogit(mu_Apun + Apun_z * sigma_Apun), dims="subj") 
            gamma = pm.Deterministic('gamma', pm.math.invlogit(mu_gamma + gamma_z * sigma_gamma) * 10.0, dims="subj") 

            rho_t = rho[self.user_codes]
            lam_t = lam[self.user_codes]
            theta_t = theta[self.user_codes]
            Arew_t = Arew[self.user_codes]
            Apun_t = Apun[self.user_codes]
            gamma_t = gamma[self.user_codes]
            
            N_idx = pt.arange(self.n_cards) 
            
            def update_step(k_t, pen_t, xg_t, xl_t, p_obj_t, rho_i, lam_i, theta_i, Arew_i, Apun_i, gamma_i, P_prev):
                # 1. Считаем EU
                u_gain = xg_t ** rho_i
                u_loss = -lam_i * (pt.abs(xl_t) ** rho_i)
                EU_t = P_prev * u_gain + (1 - P_prev) * u_loss
                
                # 2. Softmax вероятности выбора
                EU_scaled = EU_t / self.scale_factor
                exp_eu = pt.exp((EU_scaled - pt.max(EU_scaled)) * theta_i)
                prob_choice = exp_eu / pt.sum(exp_eu)
                
                # !!! ЗАЩИТА ОТ -inf В LIKELIHOOD !!!
                prob_choice = pt.clip(prob_choice, 1e-10, 1.0 - 1e-10)
                prob_choice = prob_choice / pt.sum(prob_choice)
                
                # 3. Обновление вероятностей
                p_K = p_obj_t[pt.cast(k_t, 'int32')]
                G = pt.exp(-gamma_i * pt.abs(p_obj_t - p_K))
                is_success = pt.eq(pen_t, 0)
                
                mask_le_K = pt.le(N_idx, k_t)
                mask_lt_K = pt.lt(N_idx, k_t) # Для штрафа берется строгий знак <
                
                P_succ = pt.where(mask_le_K,
                                  P_prev + Arew_i * (1 - P_prev),
                                  P_prev + Arew_i * G * (1 - P_prev))
                
                P_fail = pt.where(mask_lt_K,
                                  P_prev + Arew_i * (1 - P_prev), 
                                  P_prev + Apun_i * (0 - P_prev)) 
                                  
                P_next = pt.where(is_success, P_succ, P_fail)
                P_next = pt.clip(P_next, 0.001, 0.999) 
                
                return P_next, prob_choice

            # Явное кастование в int32 для pt.eq
            is_new_user = np.concatenate(([1], (np.diff(self.user_codes) != 0).astype(np.int32)))
            
            def scan_fn(k_t, pen_t, xg_t, xl_t, p_obj_t, rho_i, lam_i, theta_i, Arew_i, Apun_i, gamma_i, is_new, P_prev):
                P_current = pt.where(pt.eq(is_new, 1), p_obj_t, P_prev) 
                P_next, prob_choice = update_step(k_t, pen_t, xg_t, xl_t, p_obj_t, rho_i, lam_i, theta_i, Arew_i, Apun_i, gamma_i, P_current)
                return P_next, prob_choice
                
            [_, probs], _ = scan(
                fn=scan_fn,
                sequences=[pt.as_tensor(k_cards), pt.as_tensor(is_penalty), pt.as_tensor(x_gain), 
                           pt.as_tensor(x_loss), pt.as_tensor(p_obj), rho_t, lam_t, theta_t, 
                           Arew_t, Apun_t, gamma_t, pt.as_tensor(is_new_user)],
                outputs_info=[pt.as_tensor(p_obj[0]), None] 
            )
            
            pm.Categorical('obs', p=probs, observed=choices)
            
            print("[*] Граф PyTensor построен. Начинаем сэмплирование NUTS...")
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, return_inferencedata=True, progressbar=False)
            
        return self.trace, self.subj_labels

    def simulate_subject(self, params, df_subj):
        rho, lam, theta, Arew, Apun, gamma = params
        sim_choices = []
        
        P_prev = df_subj[[f'p_obj_n_{i}' for i in range(self.n_cards)]].iloc[0].values.astype(float)
        
        for idx, row in df_subj.iterrows():
            xg = row[[f'gain_n_{i}' for i in range(self.n_cards)]].values.astype(float)
            xl = row[[f'loss_n_{i}' for i in range(self.n_cards)]].values.astype(float)
            p_obj = row[[f'p_obj_n_{i}' for i in range(self.n_cards)]].values.astype(float)
            k_t = row['k_cards_drawn']
            pen_t = row['is_penalty']
            
            u_gain = xg ** rho
            u_loss = -lam * (np.abs(xl) ** rho)
            EU = P_prev * u_gain + (1 - P_prev) * u_loss
            
            EU_scaled = EU / self.scale_factor
            exp_eu = np.exp((EU_scaled - np.max(EU_scaled)) * theta)
            prob_choice = exp_eu / np.sum(exp_eu)
            
            choice = np.random.choice(np.arange(self.n_cards), p=prob_choice)
            sim_choices.append(choice)
            
            # Обновление P_prev
            P_next = np.zeros_like(P_prev)
            p_K = p_obj[int(k_t)]
            G = np.exp(-gamma * np.abs(p_obj - p_K))
            
            for N in range(self.n_cards):
                if pen_t == 0:
                    if N <= k_t:
                        P_next[N] = P_prev[N] + Arew * (1 - P_prev[N])
                    else:
                        P_next[N] = P_prev[N] + Arew * G[N] * (1 - P_prev[N])
                else:
                    if N < k_t:
                        P_next[N] = P_prev[N] + Arew * (1 - P_prev[N])
                    else:
                        P_next[N] = P_prev[N] + Apun * (0 - P_prev[N])
            P_prev = np.clip(P_next, 0.001, 0.999)
            
        df_sim = df_subj.copy()
        df_sim['choice'] = sim_choices
        return df_sim

    def posterior_predictive_check(self, trace, df, n_sims=50, save_path="ccct_v2_ppc.png"):
        print("[*] Выполнение Posterior Predictive Check (PPC)...")
        post = trace.posterior
        n_draws = post.dims['draw'] * post.dims['chain']
        rng = np.random.default_rng(42)
        sample_idxs = rng.choice(n_draws, n_sims, replace=False)
        
        all_metrics = []
        for s_idx in tqdm(sample_idxs, desc="Симуляция PPC"):
            chain = s_idx // post.dims['draw']
            draw = s_idx % post.dims['draw']
            
            sim_adv_s = []
            for u_idx, uid in enumerate(self.subj_labels):
                u_data = df[df['user_id'] == uid]
                params = (
                    post['rho'].values[chain, draw, u_idx],
                    post['lam'].values[chain, draw, u_idx],
                    post['theta'].values[chain, draw, u_idx],
                    post['Arew'].values[chain, draw, u_idx],
                    post['Apun'].values[chain, draw, u_idx],
                    post['gamma'].values[chain, draw, u_idx]
                )
                sim_df = self.simulate_subject(params, u_data)
                sim_adv_s.extend(sim_df['choice'].values)
                
            all_metrics.append(sim_adv_s)
            
        all_metrics = np.array(all_metrics) # (n_sims, total_trials)
        real_choices = df['choice'].values
        
        sim_means = np.mean(all_metrics, axis=0)
        ppp = np.mean(np.mean(all_metrics, axis=1) > np.mean(real_choices))
        r2 = r2_score(real_choices, sim_means)
        rmse = np.sqrt(mean_squared_error(real_choices, sim_means))
        mae = mean_absolute_error(real_choices, sim_means)
        hit_rate = np.mean(np.round(sim_means) == real_choices)
        
        metrics_df = pd.DataFrame([{
            'ppp': ppp, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'HitRate': hit_rate
        }])
        
        # Визуализация Timecourse
        plt.figure(figsize=(12, 6))
        plt.plot(np.mean(all_metrics, axis=0)[:100], label='Simulated (Mean)', alpha=0.8)
        plt.plot(real_choices[:100], label='Real Choices', linestyle='--')
        plt.fill_between(range(100), np.percentile(all_metrics, 2.5, axis=0)[:100], 
                         np.percentile(all_metrics, 97.5, axis=0)[:100], color='gray', alpha=0.3, label='95% HDI')
        plt.title(f"PPC Timecourse (ppp={ppp:.3f}, R2={r2:.3f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return metrics_df

    def parameter_recovery(self, df_template, n_subjects=30, n_trials=40):
        print(f"[*] Запуск Иерархического Parameter Recovery (N={n_subjects})...")
        rng = np.random.default_rng(42)
        
        # Генерация из эмпирических границ (суженных для PR)
        true_rho = rng.uniform(0.5, 2.5, n_subjects)
        true_lam = rng.uniform(1.0, 5.0, n_subjects)
        true_theta = rng.uniform(0.5, 4.0, n_subjects)
        true_Arew = rng.uniform(0.1, 0.9, n_subjects)
        true_Apun = rng.uniform(0.1, 0.9, n_subjects)
        true_gamma = rng.uniform(0.1, 3.0, n_subjects)
        
        sim_data = []
        uids = df_template['user_id'].unique()[:n_subjects]
        for idx, uid in enumerate(uids):
            subj_data = df_template[df_template['user_id'] == uid].iloc[:n_trials].copy()
            subj_data['user_id'] = f"sim_{idx}"
            params = (true_rho[idx], true_lam[idx], true_theta[idx], true_Arew[idx], true_Apun[idx], true_gamma[idx])
            df_s = self.simulate_subject(params, subj_data)
            sim_data.append(df_s)
            
        rec_df = pd.concat(sim_data, ignore_index=True)
        rec_model = CCCTModel_v2(n_cards=self.n_cards, scale_factor=self.scale_factor)
        rec_trace, rec_uids = rec_model.fit(rec_df, draws=1000, tune=1000, chains=4)
        
        # Оценка
        post = rec_trace.posterior
        metrics_list = []
        
        fit_rho = post['rho'].mean(dim=['chain', 'draw']).values
        fit_lam = post['lam'].mean(dim=['chain', 'draw']).values
        fit_theta = post['theta'].mean(dim=['chain', 'draw']).values
        
        recovery_df = pd.DataFrame({
            'user_id': rec_uids,
            'true_rho': true_rho, 'fit_rho': fit_rho,
            'true_lam': true_lam, 'fit_lam': fit_lam,
            'true_theta': true_theta, 'fit_theta': fit_theta
        })
        
        # Расчет Coverage, Bias, RMSE
        for p_name, t_vals, f_vals in zip(['rho', 'lam', 'theta'], [true_rho, true_lam, true_theta], [fit_rho, fit_lam, fit_theta]):
            hdi = az.hdi(rec_trace, var_names=[p_name])[p_name].values
            coverage = np.mean((t_vals >= hdi[:, 0]) & (t_vals <= hdi[:, 1]))
            r2 = r2_score(t_vals, f_vals)
            bias = np.mean(f_vals - t_vals)
            rmse = np.sqrt(mean_squared_error(t_vals, f_vals))
            metrics_list.append({'Parameter': p_name, 'R2': r2, 'Bias': bias, 'RMSE': rmse, 'Coverage': coverage})
            
        # Визуализация PR
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, p_name, t_vals, f_vals in zip(axes, ['rho', 'lam', 'theta'], [true_rho, true_lam, true_theta], [fit_rho, fit_lam, fit_theta]):
            sns.scatterplot(x=t_vals, y=f_vals, ax=ax)
            ax.plot([min(t_vals), max(t_vals)], [min(t_vals), max(t_vals)], 'r--')
            ax.set_title(f"{p_name} (R2={r2_score(t_vals, f_vals):.2f})")
            ax.set_xlabel('True')
            ax.set_ylabel('Recovered')
        plt.tight_layout()
        plt.savefig("ccct_v2_recovery_scatter.png", dpi=300)
        plt.close()
            
        return recovery_df, pd.DataFrame(metrics_list)