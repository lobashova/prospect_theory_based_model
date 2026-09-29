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

class IGTRelativeUncertaintyModel:
    def __init__(self, df, n_trials=100, scale_factor=100.0):
        self.df = df
        col_mapping = {
                        'deck_num': 'choice',
                        'points_earned': 'outcome'
                    }
        self.df.rename(columns={k: v for k, v in col_mapping.items() if k in self.df.columns}, inplace=True)
        self.n_trials = n_trials
        self.scale_factor = scale_factor
        
        self.users = self.df['user_id'].astype('category')
        self.user_codes = self.users.cat.codes.values
        self.n_subj = len(self.users.cat.categories)
        self.subj_labels = self.users.cat.categories.values
        
    def fit(self, draws=1500, tune=1500, chains=4, cores=4):
        print(f"[*] Инициализация HBA NUTS для {self.n_subj} участников...")
        
        choices = np.zeros((self.n_trials, self.n_subj), dtype=int)
        outcomes = np.zeros((self.n_trials, self.n_subj), dtype=float)
        
        for u_idx, uid in enumerate(self.subj_labels):
            u_data = self.df[self.df['user_id'] == uid].iloc[:self.n_trials]
            choices[:len(u_data), u_idx] = u_data['choice'].values
            outcomes[:len(u_data), u_idx] = u_data['outcome'].values / self.scale_factor
            
        coords = {"subj": self.subj_labels, "opts": np.arange(4)}
        
        with pm.Model(coords=coords) as self.model:
            # Гиперпараметры 
            mu_rho = pm.Normal('mu_rho', 0, 1)
            sigma_rho = pm.HalfNormal('sigma_rho', 1)
            mu_lam = pm.Normal('mu_lam', 0, 1)
            sigma_lam = pm.HalfNormal('sigma_lam', 1)
            mu_Arew = pm.Normal('mu_Arew', 0, 1)
            sigma_Arew = pm.HalfNormal('sigma_Arew', 1)
            mu_Apun = pm.Normal('mu_Apun', 0, 1)
            sigma_Apun = pm.HalfNormal('sigma_Apun', 1)
            mu_a_unc = pm.Normal('mu_a_unc', 0, 1)
            sigma_a_unc = pm.HalfNormal('sigma_a_unc', 1)
            mu_w = pm.Normal('mu_w', 0, 1)
            sigma_w = pm.HalfNormal('sigma_w', 1)
            mu_theta = pm.Normal('mu_theta', 0, 1)
            sigma_theta = pm.HalfNormal('sigma_theta', 1)
            
            # Z-скоры
            rho_z = pm.Normal('rho_z', 0, 1, dims="subj")
            lam_z = pm.Normal('lam_z', 0, 1, dims="subj")
            Arew_z = pm.Normal('Arew_z', 0, 1, dims="subj")
            Apun_z = pm.Normal('Apun_z', 0, 1, dims="subj")
            a_unc_z = pm.Normal('a_unc_z', 0, 1, dims="subj")
            w_z = pm.Normal('w_z', 0, 1, dims="subj")
            theta_z = pm.Normal('theta_z', 0, 1, dims="subj")
            
            # Трансформации
            rho = pm.Deterministic('rho', pm.math.invlogit(mu_rho + rho_z * sigma_rho) * 3.0, dims="subj")
            lam = pm.Deterministic('lam', pm.math.invlogit(mu_lam + lam_z * sigma_lam) * 10.0, dims="subj")
            Arew = pm.Deterministic('Arew', pm.math.invlogit(mu_Arew + Arew_z * sigma_Arew), dims="subj")
            Apun = pm.Deterministic('Apun', pm.math.invlogit(mu_Apun + Apun_z * sigma_Apun), dims="subj")
            a_unc = pm.Deterministic('a_unc', pm.math.invlogit(mu_a_unc + a_unc_z * sigma_a_unc), dims="subj")
            w = pm.Deterministic('w', (pm.math.invlogit(mu_w + w_z * sigma_w) * 20.0) - 10.0, dims="subj") 
            theta = pm.Deterministic('theta', pm.math.invlogit(mu_theta + theta_z * sigma_theta) * 10.0, dims="subj")
            
            def step(c_t, o_t, u_g_prev, u_l_prev, ef_prev, l_prev, rho, lam, Arew, Apun, a_unc, w):
                c_mat = pt.eq(pt.arange(4)[None, :], c_t[:, None])
                is_pos = pt.ge(o_t, 0)
                
                u_val = pt.switch(is_pos, o_t ** rho, -lam * (pt.abs(o_t) ** rho))
                
                u_g_new = pt.switch(c_mat & is_pos[:, None], u_val[:, None], u_g_prev)
                u_l_new = pt.switch(c_mat & ~is_pos[:, None], u_val[:, None], u_l_prev)
                
                ef_chosen = pt.sum(ef_prev * c_mat, axis=1)
                ef_upd = pt.switch(is_pos, ef_chosen + Arew * (1 - ef_chosen), ef_chosen - Apun * ef_chosen)
                ef_new = pt.switch(c_mat, ef_upd[:, None], ef_prev)
                
                l_chosen = pt.sum(l_prev * c_mat, axis=1)
                l_upd = l_chosen + (1 - l_chosen) * a_unc
                l_new = pt.switch(c_mat, l_upd[:, None], l_prev)
                
                # ИСПРАВЛЕНИЕ: Защита от деления на ноль в PyTensor
                l_inv = pt.clip(1.0 - l_new, 1e-10, 1.0)
                U_new = l_inv / pt.sum(l_inv, axis=1, keepdims=True)
                
                EU_new = ef_new * u_g_new + (1 - ef_new) * u_l_new
                V_new = EU_new + w[:, None] * U_new
                
                return u_g_new, u_l_new, ef_new, l_new, V_new

            u_g_0 = pt.zeros((self.n_subj, 4))
            u_l_0 = pt.zeros((self.n_subj, 4))
            ef_0 = pt.ones((self.n_subj, 4)) * 0.5
            l_0 = pt.zeros((self.n_subj, 4))
            V_0 = pt.ones((self.n_subj, 4)) * (w[:, None] * 0.25)
            
            [_, _, _, _, V_t], _ = scan(
                fn=step,
                sequences=[pt.as_tensor_variable(choices), pt.as_tensor_variable(outcomes)],
                outputs_info=[u_g_0, u_l_0, ef_0, l_0, None],
                non_sequences=[rho, lam, Arew, Apun, a_unc, w],
                strict=True
            )
            
            V_for_choice = pt.concatenate([V_0[None, :, :], V_t[:-1]], axis=0)
            V_scaled = V_for_choice * theta[None, :, None]
            
            pi = pm.math.softmax(V_scaled, axis=2)
            
            # ИСПРАВЛЕНИЕ: Жесткое отсечение нулей, чтобы лог-правдоподобие не выдавало -inf
            pi = pt.clip(pi, 1e-10, 1.0 - 1e-10)
            pi = pi / pt.sum(pi, axis=2, keepdims=True)
            
            pm.Categorical('obs', p=pi, observed=choices)
            
            trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, return_inferencedata=True, progressbar=False)
            self.trace = trace
            return trace, self.subj_labels

    def simulate_subject(self, params, df_template):
        rho, lam, Arew, Apun, a_unc, w, theta = params
        n_t = len(df_template)
        
        u_g = np.zeros(4)
        u_l = np.zeros(4)
        ef = np.ones(4) * 0.5
        l_arr = np.zeros(4)
        
        sim_choices = []
        sim_outcomes = []
        is_adv = []
        
        for t in range(n_t):
            # ИСПРАВЛЕНИЕ: Защита от деления на ноль в Numpy
            l_inv = np.clip(1.0 - l_arr, 1e-10, 1.0)
            U = l_inv / np.sum(l_inv)
            
            EU = ef * u_g + (1 - ef) * u_l
            V = EU + w * U
            
            # Softmax с защитой
            ev_scaled = V * theta
            ev_scaled -= np.nanmax(ev_scaled)
            exp_ev = np.exp(ev_scaled)
            probs = exp_ev / np.sum(exp_ev)
            
            # Дополнительный fallback, если экстремальные параметры вызвали NaN
            if np.any(np.isnan(probs)):
                probs = np.ones(4) / 4.0
            
            choice = np.random.choice(4, p=probs)
            
            row = df_template.iloc[t]
            outcome = row['outcome'] / self.scale_factor 
            
            sim_choices.append(choice)
            sim_outcomes.append(outcome * self.scale_factor)
            is_adv.append(1 if choice in [2, 3] else 0)
            
            if outcome >= 0:
                u_g[choice] = outcome ** rho
                ef[choice] = ef[choice] + Arew * (1 - ef[choice])
            else:
                u_l[choice] = -lam * (abs(outcome) ** rho)
                ef[choice] = ef[choice] - Apun * ef[choice]
                
            l_arr[choice] = l_arr[choice] + (1 - l_arr[choice]) * a_unc
            
        df_sim = df_template.copy()
        df_sim['choice'] = sim_choices
        df_sim['outcome'] = sim_outcomes
        df_sim['is_adv'] = is_adv
        return df_sim

    def posterior_predictive_check(self, trace, df, block_size=30, n_sims=100):
        print("\n[*] Запуск Posterior Predictive Check...")
        n_blocks = self.n_trials // block_size
        post = trace.posterior
        
        real_adv = np.zeros((self.n_subj, n_blocks))
        for i, uid in enumerate(self.subj_labels):
            u_data = df[df['user_id'] == uid].iloc[:self.n_trials]
            adv = u_data['choice'].isin([2, 3]).astype(int).values
            for b in range(n_blocks):
                # ИСПРАВЛЕНИЕ 1: Защита от пустых срезов при недостатке триалов
                slice_adv = adv[b*block_size : (b+1)*block_size]
                real_adv[i, b] = np.mean(slice_adv) if len(slice_adv) > 0 else np.nan
        
        rng = np.random.default_rng(42)
        total_samples = post['rho'].shape[0] * post['rho'].shape[1]
        sample_indices = rng.choice(total_samples, size=n_sims, replace=False)
        
        rho_flat = post['rho'].values.reshape(-1, self.n_subj)
        lam_flat = post['lam'].values.reshape(-1, self.n_subj)
        arew_flat = post['Arew'].values.reshape(-1, self.n_subj)
        apun_flat = post['Apun'].values.reshape(-1, self.n_subj)
        aunc_flat = post['a_unc'].values.reshape(-1, self.n_subj)
        w_flat = post['w'].values.reshape(-1, self.n_subj)
        theta_flat = post['theta'].values.reshape(-1, self.n_subj)
        
        sim_block_means = np.zeros((n_sims, n_blocks))
        
        for s_idx, t_idx in enumerate(tqdm(sample_indices, desc="Симуляция PPC")):
            sim_adv_s = np.zeros((self.n_subj, n_blocks))
            for u_idx, uid in enumerate(self.subj_labels):
                p_tuple = (rho_flat[t_idx, u_idx], lam_flat[t_idx, u_idx], arew_flat[t_idx, u_idx],
                           apun_flat[t_idx, u_idx], aunc_flat[t_idx, u_idx], w_flat[t_idx, u_idx], theta_flat[t_idx, u_idx])
                
                sim_df = self.simulate_subject(p_tuple, df[df['user_id'] == uid].iloc[:self.n_trials])
                adv = sim_df['is_adv'].values
                for b in range(n_blocks):
                    # ИСПРАВЛЕНИЕ 2: Защита от пустых срезов в симуляции
                    slice_adv = adv[b*block_size : (b+1)*block_size]
                    sim_adv_s[u_idx, b] = np.mean(slice_adv) if len(slice_adv) > 0 else np.nan
            
            # ИСПРАВЛЕНИЕ 3: Игнорируем NaN при усреднении по участникам
            sim_block_means[s_idx] = np.nanmean(sim_adv_s, axis=0)
            
        real_block_mean = np.nanmean(real_adv, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, n_blocks + 1), real_block_mean, 'r-o', label='Real Data', linewidth=2)
        plt.plot(range(1, n_blocks + 1), np.nanmean(sim_block_means, axis=0), 'b--s', label='Simulated Data', linewidth=2)
        plt.fill_between(range(1, n_blocks + 1), np.nanpercentile(sim_block_means, 2.5, axis=0), 
                         np.nanpercentile(sim_block_means, 97.5, axis=0), color='b', alpha=0.2, label='95% HDI')
        plt.xlabel('Blocks')
        plt.ylabel('Proportion of Advantageous Choices')
        plt.title('PPC: IGT Relative Uncertainty Model')
        plt.legend()
        plt.savefig("igt_rel_unc_ppc.png", dpi=300)
        plt.close()
        
        sim_mean_overall = np.nanmean(sim_block_means, axis=0)
        
        # ИСПРАВЛЕНИЕ 4: Маскируем NaN для sklearn метрик
        valid_mask = ~np.isnan(real_block_mean) & ~np.isnan(sim_mean_overall)
        
        if np.sum(valid_mask) == 0:
            print("[!] Нет данных для расчета метрик.")
            return pd.DataFrame()

        rmse = np.sqrt(mean_squared_error(real_block_mean[valid_mask], sim_mean_overall[valid_mask]))
        mae = mean_absolute_error(real_block_mean[valid_mask], sim_mean_overall[valid_mask])
        r2 = r2_score(real_block_mean[valid_mask], sim_mean_overall[valid_mask])
        ppp = np.mean(np.nanmean(sim_block_means[:, valid_mask], axis=1) > np.mean(real_block_mean[valid_mask]))
        
        metrics_df = pd.DataFrame([{'Model': 'IGT_Relative_Uncertainty', 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'ppp': ppp}])
        return metrics_df

    def parameter_recovery(self, n_subjects=30, n_trials=150, save_prefix="igt_rel_unc_recovery"):
        print(f"[*] Запуск HBA Parameter Recovery (N={n_subjects})...")
        if getattr(self, 'trace', None) is None:
            raise ValueError("Сначала обучите модель на реальных данных для извлечения эмпирических параметров!")
            
        post = self.trace.posterior
        rng = np.random.default_rng(42)
        
        # 1. Извлекаем правильные параметры для IGTRelativeUncertaintyModel
        true_params = {
            'rho': rng.choice(post['rho'].mean(dim=["chain", "draw"]).values, n_subjects),
            'lam': rng.choice(post['lam'].mean(dim=["chain", "draw"]).values, n_subjects),
            'Arew': rng.choice(post['Arew'].mean(dim=["chain", "draw"]).values, n_subjects),
            'Apun': rng.choice(post['Apun'].mean(dim=["chain", "draw"]).values, n_subjects),
            'a_unc': rng.choice(post['a_unc'].mean(dim=["chain", "draw"]).values, n_subjects), # Исправлено
            'w': rng.choice(post['w'].mean(dim=["chain", "draw"]).values, n_subjects),         # Исправлено
            'theta': rng.choice(post['theta'].mean(dim=["chain", "draw"]).values, n_subjects),
        }
        
        # 2. Симулируем данные
        sim_data = []
        for i in range(n_subjects):
            # Передаем правильные аргументы в simulate_subject
            df = self.simulate_subject(
                (true_params['rho'][i], true_params['lam'][i], true_params['Arew'][i],
                true_params['Apun'][i], true_params['a_unc'][i], true_params['w'][i],
                true_params['theta'][i]), 
                self.df[self.df['user_id'] == self.subj_labels[0]].iloc[:n_trials] # Берем любой шаблон
            )
            
            # (Здесь оставляем ваш код генерации выплат outcomes)
            outcomes = []
            for c in df['choice']:
                if c == 0:   # Колода A
                    win, loss = 100, -250 if rng.random() < 0.5 else 0
                elif c == 1: # Колода B
                    win, loss = 100, -625 if rng.random() < 0.2 else 0
                elif c == 2: # Колода C
                    win, loss = 50, -50 if rng.random() < 0.5 else 0
                elif c == 3: # Колода D
                    win, loss = 50, -125 if rng.random() < 0.2 else 0
                else:
                    win, loss = 0, 0
                outcomes.append(win + loss)
                
            df['outcome'] = outcomes
            df['user_id'] = f'sim_{i}' # Убедитесь, что колонка называется user_id, как ожидает fit()
            sim_data.append(df)
            
        sim_df = pd.concat(sim_data).reset_index(drop=True)
        
        # 3. Подгонка правильной модели
        rec_model = IGTRelativeUncertaintyModel(sim_df, n_trials=n_trials, scale_factor=self.scale_factor)
        rec_trace, _ = rec_model.fit(draws=500, tune=500, chains=2)
        rec_post = rec_trace.posterior
        
        # 4. Расчет метрик с правильным списком параметров
        metrics = []
        recovery_dict = {'user_id': [f'sim_{i}' for i in range(n_subjects)]}
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Заменили K_prime и beta_p на a_unc и w
        params_list = ['rho', 'lam', 'Arew', 'Apun', 'a_unc', 'w', 'theta']
        
        for idx, param in enumerate(params_list):
            t_val = true_params[param]
            f_val = rec_post[param].mean(dim=["chain", "draw"]).values
            hdi = az.hdi(rec_trace, var_names=[param], hdi_prob=0.95)[param].values
            
            # Сохранение параметров субъекта
            recovery_dict[f'true_{param}'] = t_val
            recovery_dict[f'fit_{param}'] = f_val
            
            cov_arr = (t_val >= hdi[:, 0]) & (t_val <= hdi[:, 1])
            recovery_dict[f'coverage_{param}'] = cov_arr.astype(int)
            
            # Агрегированные метрики
            r = np.corrcoef(t_val, f_val)[0, 1] if np.std(f_val) > 0 else 0
            rmse = np.sqrt(mean_squared_error(t_val, f_val))
            bias = np.mean(f_val - t_val)
            coverage = np.mean(cov_arr)
            
            metrics.append({'Param': param, 'r': r, 'RMSE': rmse, 'Bias': bias, 'Coverage': coverage})
            
            # Отрисовка скаттерплота
            sns.scatterplot(x=t_val, y=f_val, ax=axes[idx], color='teal', alpha=0.8)
            min_v, max_v = min(t_val.min(), f_val.min()), max(t_val.max(), f_val.max())
            axes[idx].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label="Perfect Recovery")
            axes[idx].set_title(f"{param}\nr={r:.2f}, Cov={coverage:.2f}")
            axes[idx].set_xlabel("True")
            axes[idx].set_ylabel("Recovered")
            
        # Удаляем лишний (8-й) график
        if len(params_list) < len(axes):
            fig.delaxes(axes[-1])
            
        plt.tight_layout()
        
        # 5. СОХРАНЕНИЕ
        plt.savefig(f"{save_prefix}_plots.png", dpi=300)
        plt.close()
        
        metrics_df = pd.DataFrame(metrics)
        recovery_df = pd.DataFrame(recovery_dict)
        
        recovery_df.to_csv(f"{save_prefix}_params.csv", index=False)
        metrics_df.to_csv(f"{save_prefix}_metrics.csv", index=False)
        
        print(f"[✓] Графики сохранены в {save_prefix}_plots.png")
        print(f"[✓] Параметры сохранены в {save_prefix}_params.csv")
        print(f"[✓] Метрики сохранены в {save_prefix}_metrics.csv")
        
        print("\nМетрики Parameter Recovery:")
        print(metrics_df)
        
        return recovery_df, metrics_df