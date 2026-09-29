import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.special import expit
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

class IGTSequentialV2Model:
    def __init__(self, data, scale_factor=100.0, n_trials=150):
        self.data = data
        col_mapping = {
                    'deck_num': 'choice',
                    'points_earned': 'outcome'
                }
        self.data.rename(columns={k: v for k, v in col_mapping.items() if k in self.data.columns}, inplace=True)
        self.scale_factor = scale_factor
        self.n_decks = 4
        self.max_trials = n_trials

        self.users = self.data['user_id'].astype('category')
        self.user_codes = self.users.cat.codes.values
        self.n_subj = len(self.users.cat.categories)
        self.subj_labels = self.users.cat.categories.values
        
        # Подготовка матриц (пользователь x триал)
        self.choice_mat = np.zeros((self.n_subj, self.max_trials), dtype=int)
        self.outcome_mat = np.zeros((self.n_subj, self.max_trials))
        self.mask_mat = np.zeros((self.n_subj, self.max_trials), dtype=bool)
        
        for i, uid in enumerate(self.subj_labels):
            user_df = self.data[self.data['user_id'] == uid]
            n_t = min(len(user_df), self.max_trials)
            self.choice_mat[i, :n_t] = user_df['choice'].values[:n_t]
            # Используем net outcome (чистый исход), как требует формула
            self.outcome_mat[i, :n_t] = user_df['outcome'].values[:n_t]
            self.mask_mat[i, :n_t] = True

        self.model = None
        self.trace = None

    def build_model(self):
        coords = {"subj": self.subj_labels, "decks": np.arange(self.n_decks)}
        with pm.Model(coords=coords) as self.model:
            # Гиперпараметры (Non-centered parameterization)
            mu = pm.Normal('mu', mu=0, sigma=1, shape=7)
            sigma = pm.HalfNormal('sigma', sigma=1, shape=7)
            
            z = pm.Normal('z', mu=0, sigma=1, shape=(7, self.n_subj))
            
            # Трансформации с классическими границами
            rho = pm.Deterministic('rho', pm.math.invlogit(mu[0] + z[0] * sigma[0]) * 3.0, dims="subj")
            lam = pm.Deterministic('lam', pm.math.invlogit(mu[1] + z[1] * sigma[1]) * 10.0, dims="subj")
            Arew = pm.Deterministic('Arew', pm.math.invlogit(mu[2] + z[2] * sigma[2]), dims="subj")
            Apun = pm.Deterministic('Apun', pm.math.invlogit(mu[3] + z[3] * sigma[3]), dims="subj")
            alpha = pm.Deterministic('alpha', pm.math.invlogit(mu[4] + z[4] * sigma[4]), dims="subj")
            phi = pm.Deterministic('phi', pm.math.invlogit(mu[5] + z[5] * sigma[5]) * 10.0, dims="subj")
            theta = pm.Deterministic('theta', pm.math.invlogit(mu[6] + z[6] * sigma[6]) * 10.0, dims="subj")

            choice_data = pt.as_tensor_variable(self.choice_mat.T)
            outcome_data = pt.as_tensor_variable(self.outcome_mat.T / self.scale_factor)

            def step_func(choice_t, outcome_t, EF_prev, Explore_prev, u_gain_prev, u_loss_prev, 
                          rho, lam, Arew, Apun, alpha, phi):
                
                # Ожидаемая полезность: EU = EF * u_gain + (1 - EF) * u_loss
                EU = EF_prev * u_gain_prev + (1.0 - EF_prev) * u_loss_prev
                # Интеграция V = EU + Explore
                V = EU + Explore_prev

                is_win = pt.ge(outcome_t, 0.0)
                
                # Обновление ожидаемой частоты наград (Дельта-правило)
                new_EF_c = pt.where(
                    is_win, 
                    EF_prev[pt.arange(self.n_subj), choice_t] + Arew * (1.0 - EF_prev[pt.arange(self.n_subj), choice_t]), 
                    EF_prev[pt.arange(self.n_subj), choice_t] - Apun * EF_prev[pt.arange(self.n_subj), choice_t]
                )
                EF_new = pt.set_subtensor(EF_prev[pt.arange(self.n_subj), choice_t], new_EF_c)

                # Обновление функции полезности исходов
                new_ugain_c = pt.where(is_win, pt.power(outcome_t, rho), u_gain_prev[pt.arange(self.n_subj), choice_t])
                u_gain_new = pt.set_subtensor(u_gain_prev[pt.arange(self.n_subj), choice_t], new_ugain_c)

                new_uloss_c = pt.where(pt.lt(outcome_t, 0.0), -lam * pt.power(pt.abs(outcome_t), rho), u_loss_prev[pt.arange(self.n_subj), choice_t])
                u_loss_new = pt.set_subtensor(u_loss_prev[pt.arange(self.n_subj), choice_t], new_uloss_c)

                # Обновление бонуса за исследование
                Explore_unchozen = Explore_prev + alpha[:, None] * (phi[:, None] - Explore_prev)
                Explore_new = pt.set_subtensor(Explore_unchozen[pt.arange(self.n_subj), choice_t], 0.0)

                return V, EF_new, Explore_new, u_gain_new, u_loss_new

            # Инициализация
            EF_init = pt.ones((self.n_subj, self.n_decks)) * 0.5
            Explore_init = pt.ones((self.n_subj, self.n_decks))
            u_gain_init = pt.zeros((self.n_subj, self.n_decks))
            u_loss_init = pt.zeros((self.n_subj, self.n_decks))

            [V_vals, _, _, _, _], _ = scan(
                fn=step_func,
                sequences=[choice_data, outcome_data],
                outputs_info=[None, EF_init, Explore_init, u_gain_init, u_loss_init],
                non_sequences=[rho, lam, Arew, Apun, alpha, phi],
                strict=True
            )

            # Softmax с защитой от переполнения
            V_scaled = V_vals * theta[:, None]
            probs = pt.exp(V_scaled - pt.max(V_scaled, axis=2, keepdims=True))
            probs = probs / pt.sum(probs, axis=2, keepdims=True)
            probs_clipped = pt.clip(probs, 1e-6, 1.0 - 1e-6)

            # Правдоподобие
            logp_step = pt.log(probs_clipped[pt.arange(self.max_trials)[:, None], pt.arange(self.n_subj), choice_data])
            pm.Potential("loglike", pt.sum(logp_step * self.mask_mat.T))

    def fit(self, draws=1500, tune=1500, chains=4, cores=4):
        print(f"[*] Подгонка HBA NUTS для {self.n_subj} участников...")
        if self.model is None:
            self.build_model()
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, target_accept=0.95, return_inferencedata=True)
        return self.trace, self.subj_labels

    def simulate_subject(self, params, n_trials=150):
        rho, lam, Arew, Apun, alpha, phi, theta = params
        sim_choices = []
        
        EF = np.ones(4) * 0.5
        Explore = np.ones(4)
        u_gain = np.zeros(4)
        u_loss = np.zeros(4)

        for t in range(n_trials):
            EU = EF * u_gain + (1.0 - EF) * u_loss
            V = EU + Explore
            
            V_scaled = V * theta
            V_scaled -= np.max(V_scaled)
            ex = np.exp(V_scaled)
            probs = ex / np.sum(ex)
            
            c = np.random.choice(4, p=probs)
            sim_choices.append(c)
            
            # Кастомная схема выплат и вероятностей IGT (масштабированная)
            if c == 0:
                win, loss = 1.0, -2.5 if np.random.rand() < 0.5 else 0.0
            elif c == 1:
                win, loss = 1.0, -6.25 if np.random.rand() < 0.2 else 0.0
            elif c == 2:
                win, loss = 0.5, -0.5 if np.random.rand() < 0.5 else 0.0
            else: # c == 3
                win, loss = 0.5, -1.25 if np.random.rand() < 0.2 else 0.0
                
            x_val = win + loss
            
            # Обновления
            if x_val >= 0:
                EF[c] += Arew * (1.0 - EF[c])
                u_gain[c] = (x_val) ** rho
            else:
                EF[c] -= Apun * EF[c]
                u_loss[c] = -lam * (abs(x_val) ** rho)
                
            Explore = Explore + alpha * (phi - Explore)
            Explore[c] = 0.0

        return np.array(sim_choices)

    def posterior_predictive_check(self, trace, df, n_sims=100, block_size=30, save_path="igt_seq_ppc.png"):
        print("[*] Запуск Posterior Predictive Check...")
        post = trace.posterior
        n_blocks = self.max_trials // block_size
        
        # 1. Считаем реальную кривую научения по блокам
        real_adv = np.zeros((self.n_subj, n_blocks))
        for i, uid in enumerate(self.subj_labels):
            u_data = df[df['user_id'] == uid]
            adv = u_data['choice'].isin([2, 3]).astype(int).values
            for b in range(n_blocks):
                block_data = adv[b*block_size:(b+1)*block_size]
                if len(block_data) > 0:
                    real_adv[i, b] = np.mean(block_data)
                else:
                    real_adv[i, b] = np.nan
        
        # Усредненная реальная кривая и скаляр для PPP
        real_mean_curve = np.nanmean(real_adv, axis=0) # [5 блоков]
        real_scalar_mean = np.nanmean(real_mean_curve)
        
        # 2. Подготовка симуляций
        sim_adv_means = np.zeros((n_sims, n_blocks))
        sim_scalar_means = np.zeros(n_sims)
        
        rng = np.random.default_rng(42)
        total_samples = post.dims['chain'] * post.dims['draw']
        sample_indices = rng.choice(total_samples, size=n_sims, replace=False)

        rho_s = post['rho'].values.reshape(-1, self.n_subj)
        lam_s = post['lam'].values.reshape(-1, self.n_subj)
        Arew_s = post['Arew'].values.reshape(-1, self.n_subj)
        Apun_s = post['Apun'].values.reshape(-1, self.n_subj)
        alpha_s = post['alpha'].values.reshape(-1, self.n_subj)
        phi_s = post['phi'].values.reshape(-1, self.n_subj)
        theta_s = post['theta'].values.reshape(-1, self.n_subj)

        # 3. Запуск симуляций
        for s_idx, idx in enumerate(tqdm(sample_indices, desc="Симуляция PPC")):
            sim_adv_s = np.zeros((self.n_subj, n_blocks))
            for i, uid in enumerate(self.subj_labels):
                params = (rho_s[idx, i], lam_s[idx, i], Arew_s[idx, i], Apun_s[idx, i], alpha_s[idx, i], phi_s[idx, i], theta_s[idx, i])
                sim_c = self.simulate_subject(params, self.max_trials)
                sim_adv = np.isin(sim_c, [2, 3]).astype(int)
                for b in range(n_blocks):
                    sim_adv_s[i, b] = np.mean(sim_adv[b*block_size:(b+1)*block_size])
            
            # Средняя кривая для одной симуляции
            sim_curve = np.nanmean(sim_adv_s, axis=0)
            sim_adv_means[s_idx] = sim_curve
            sim_scalar_means[s_idx] = np.nanmean(sim_curve)

        # 4. Агрегация результатов
        sim_mean_curve = np.nanmean(sim_adv_means, axis=0)
        hdi = az.hdi(sim_adv_means)

        # Отфильтруем пустые блоки, если есть NaN
        valid_mask = ~np.isnan(real_mean_curve)
        real_valid = real_mean_curve[valid_mask]
        sim_valid = sim_mean_curve[valid_mask]

        # Глобальные метрики по кривой (сравниваем вектор из 5 блоков)
        r2 = r2_score(real_valid, sim_valid) if len(real_valid) > 1 else np.nan
        rmse = np.sqrt(mean_squared_error(real_valid, sim_valid))
        mae = mean_absolute_error(real_valid, sim_valid)
        
        # Bayesian p-value (Глобальный, на основе общих средних)
        ppp = np.mean(sim_scalar_means >= real_scalar_mean)
        
        metrics_list = [{
            'Model': 'IGTSequentialV2',
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'ppp': ppp
        }]

        # 5. Построение графика
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_blocks + 1), real_mean_curve, 'r-o', label='Реальные данные', linewidth=2)
        plt.plot(range(1, n_blocks + 1), sim_mean_curve, 'b--', label='Симуляция (Среднее)', linewidth=2)
        plt.fill_between(range(1, n_blocks + 1), hdi[:, 0], hdi[:, 1], color='blue', alpha=0.2, label='95% HDI')
        plt.xlabel('Блок триалов')
        plt.ylabel('Доля выбора выгодных колод')
        plt.title('PPC: Доля выгодных колод во времени (IGT Seq)')
        plt.legend()
        plt.savefig(save_path, dpi=300)
        plt.close()

        return pd.DataFrame(metrics_list)

    def parameter_recovery(self, n_subjects=20, n_trials=150, save_path="igt_seq_recovery.png"):
        print(f"[*] Запуск Эмпирического Parameter Recovery (N={n_subjects})...")
        
        if getattr(self, 'trace', None) is None:
            raise ValueError("Сначала запустите fit() на реальных данных для извлечения эмпирических параметров!")
            
        rng = np.random.default_rng(42)
        post = self.trace.posterior
        
        # Эмпирическая генерация: выбираем параметры из средних апостериорных значений реальных испытуемых
        true_rho = rng.choice(post['rho'].mean(dim=["chain", "draw"]).values, n_subjects)
        true_lam = rng.choice(post['lam'].mean(dim=["chain", "draw"]).values, n_subjects)
        true_Arew = rng.choice(post['Arew'].mean(dim=["chain", "draw"]).values, n_subjects)
        true_Apun = rng.choice(post['Apun'].mean(dim=["chain", "draw"]).values, n_subjects)
        true_alpha = rng.choice(post['alpha'].mean(dim=["chain", "draw"]).values, n_subjects)
        true_phi = rng.choice(post['phi'].mean(dim=["chain", "draw"]).values, n_subjects)
        true_theta = rng.choice(post['theta'].mean(dim=["chain", "draw"]).values, n_subjects)

        sim_data = []
        for i in range(n_subjects):
            params = (true_rho[i], true_lam[i], true_Arew[i], true_Apun[i], true_alpha[i], true_phi[i], true_theta[i])
            sim_c = self.simulate_subject(params, n_trials)
            
            for t, c in enumerate(sim_c):
                # Обновленная схема выплат
                if c == 0:   # Колода A
                    win, loss = 100, -250 if np.random.rand() < 0.5 else 0
                elif c == 1: # Колода B
                    win, loss = 100, -625 if np.random.rand() < 0.2 else 0
                elif c == 2: # Колода C
                    win, loss = 50, -50 if np.random.rand() < 0.5 else 0
                else:        # Колода D
                    win, loss = 50, -125 if np.random.rand() < 0.2 else 0
                
                sim_data.append({'user_id': f'sim_{i}', 'trial': t, 'choice': c, 'outcome': win + loss})
        
        rec_df = pd.DataFrame(sim_data)
        rec_model = IGTSequentialV2Model(rec_df, n_trials=n_trials)
        
        # Подгонка симулированных данных (MCMC)
        rec_trace, _ = rec_model.fit(draws=1000, tune=1000, chains=4)

        rec_post = rec_trace.posterior
        metrics_list = []
        
        # Инициализируем словарь для сохранения параметров каждого субъекта
        recovery_dict = {'user_id': [f'sim_{i}' for i in range(n_subjects)]}
        
        params_dict = {
            'rho': true_rho, 'lam': true_lam, 'Arew': true_Arew, 
            'Apun': true_Apun, 'alpha': true_alpha, 'phi': true_phi, 'theta': true_theta
        }
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, (p_name, t_vals) in enumerate(params_dict.items()):
            f_vals = rec_post[p_name].mean(dim=['chain', 'draw']).values
            hdi = az.hdi(rec_trace, var_names=[p_name])[p_name].values
            
            # Сохраняем true/fit значения и попадание в интервал (coverage)
            recovery_dict[f'true_{p_name}'] = t_vals
            recovery_dict[f'fit_{p_name}'] = f_vals
            
            cov_arr = (t_vals >= hdi[:, 0]) & (t_vals <= hdi[:, 1])
            recovery_dict[f'coverage_{p_name}'] = cov_arr.astype(int)
            
            coverage = np.mean(cov_arr)
            r2 = r2_score(t_vals, f_vals)
            bias = np.mean(f_vals - t_vals)
            rmse = np.sqrt(mean_squared_error(t_vals, f_vals))
            mae = mean_absolute_error(t_vals, f_vals)
            
            # Метрика Hit Rate
            hit_rate = np.mean(np.sign(f_vals - np.mean(f_vals)) == np.sign(t_vals - np.mean(t_vals)))

            metrics_list.append({
                'Parameter': p_name, 'R2': r2, 'Bias': bias, 
                'RMSE': rmse, 'MAE': mae, 'HitRate': hit_rate, 'Coverage': coverage
            })
            
            # Визуализация
            sns.scatterplot(x=t_vals, y=f_vals, ax=axes[idx], color='teal', alpha=0.8)
            min_val, max_val = min(t_vals.min(), f_vals.min()), max(t_vals.max(), f_vals.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[idx].set_title(f"{p_name} (R2={r2:.2f}, Cov={coverage:.2f})")
            axes[idx].set_xlabel('True')
            axes[idx].set_ylabel('Recovered')
        
        # Удаляем 8-й пустой график
        if len(params_dict) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
            
        metrics_df = pd.DataFrame(metrics_list)
        recovery_df = pd.DataFrame(recovery_dict)
        
        # Гарантированное сохранение таблиц на диск
        base_path = save_path.rsplit('.', 1)[0]
        recovery_df.to_csv(f"{base_path}_params.csv", index=False)
        metrics_df.to_csv(f"{base_path}_metrics.csv", index=False)
        
        return recovery_df, metrics_df