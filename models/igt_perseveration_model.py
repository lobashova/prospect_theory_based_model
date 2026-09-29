import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
from tqdm.auto import tqdm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class IGTPerseverationModel:
    def __init__(self, data_df):
        """
        Инициализация модели IGT с персеверацией.
        Ожидается DataFrame. Автоматически адаптируется под колонки: 
        'user_id', 'trial_number', 'deck_num' (0-3), 'points_earned'.
        """
        self.data = data_df.copy()
        
        col_mapping = {
            'user_id': 'subj_id',
            'trial_number': 'trial',
            'deck_num': 'choice',
            'points_earned': 'outcome'
        }
        self.data.rename(columns={k: v for k, v in col_mapping.items() if k in self.data.columns}, inplace=True)
        
        # Масштабируем награды для избежания переполнения (Softmax overflow)
        self.data['outcome_scaled'] = self.data['outcome'] / 100.0
        
        self.subjects = self.data['subj_id'].unique()
        self.n_subj = len(self.subjects)
        self.subj_idx = {subj: i for i, subj in enumerate(self.subjects)}
        self.data['subj_idx'] = self.data['subj_id'].map(self.subj_idx)
        
        self.n_trials = self.data.groupby('subj_id').size().max()
        
        self.choices = np.zeros((self.n_trials, self.n_subj), dtype=int)
        self.outcomes = np.zeros((self.n_trials, self.n_subj), dtype=float)
        self.mask = np.zeros((self.n_trials, self.n_subj), dtype=int)
        
        for s_idx, subj in enumerate(self.subjects):
            subj_data = self.data[self.data['subj_idx'] == s_idx]
            n_t = len(subj_data)
            self.choices[:n_t, s_idx] = subj_data['choice'].values
            self.outcomes[:n_t, s_idx] = subj_data['outcome_scaled'].values
            self.mask[:n_t, s_idx] = 1

    def build_model(self):
        with pm.Model() as self.model:
            # Гиперпараметры (групповой уровень)
            mu_rho = pm.Normal('mu_rho', mu=0, sigma=1)
            mu_lam = pm.Normal('mu_lam', mu=0, sigma=1)
            mu_Arew = pm.Normal('mu_Arew', mu=0, sigma=1)
            mu_Apun = pm.Normal('mu_Apun', mu=0, sigma=1)
            mu_K_prime = pm.Normal('mu_K_prime', mu=0, sigma=1)
            mu_beta_p = pm.Normal('mu_beta_p', mu=0, sigma=1)
            mu_theta = pm.Normal('mu_theta', mu=0, sigma=1)
            
            sigma_rho = pm.HalfNormal('sigma_rho', sigma=1)
            sigma_lam = pm.HalfNormal('sigma_lam', sigma=1)
            sigma_Arew = pm.HalfNormal('sigma_Arew', sigma=1)
            sigma_Apun = pm.HalfNormal('sigma_Apun', sigma=1)
            sigma_K_prime = pm.HalfNormal('sigma_K_prime', sigma=1)
            sigma_beta_p = pm.HalfNormal('sigma_beta_p', sigma=1)
            sigma_theta = pm.HalfNormal('sigma_theta', sigma=1)
            
            # Индивидуальные параметры (с трансформацией в границы)
            # rho: (0.01, 3.0), lambda: (0.01, 10.0), Arew/Apun: (0, 1), K': (0, 5)[cite: 1]
            rho = pm.Deterministic('rho', 0.01 + 2.99 * pm.math.invlogit(pm.Normal('z_rho', mu=mu_rho, sigma=sigma_rho, shape=self.n_subj)))
            lam = pm.Deterministic('lam', 0.01 + 9.99 * pm.math.invlogit(pm.Normal('z_lam', mu=mu_lam, sigma=sigma_lam, shape=self.n_subj)))
            Arew = pm.Deterministic('Arew', pm.math.invlogit(pm.Normal('z_Arew', mu=mu_Arew, sigma=sigma_Arew, shape=self.n_subj)))
            Apun = pm.Deterministic('Apun', pm.math.invlogit(pm.Normal('z_Apun', mu=mu_Apun, sigma=sigma_Apun, shape=self.n_subj)))
            K_prime = pm.Deterministic('K_prime', 5.0 * pm.math.invlogit(pm.Normal('z_K_prime', mu=mu_K_prime, sigma=sigma_K_prime, shape=self.n_subj)))
            beta_p = pm.Deterministic('beta_p', pm.Normal('z_beta_p', mu=mu_beta_p, sigma=sigma_beta_p, shape=self.n_subj))
            theta = pm.Deterministic('theta', 10.0 * pm.math.invlogit(pm.Normal('z_theta', mu=mu_theta, sigma=sigma_theta, shape=self.n_subj)))
            
            K_val = pm.Deterministic('K_val', pt.power(3.0, K_prime) - 1.0)

            u_gain_init = pt.zeros((self.n_subj, 4), dtype='float64')
            u_loss_init = pt.zeros((self.n_subj, 4), dtype='float64')
            EF_init = pt.fill(pt.zeros((self.n_subj, 4), dtype='float64'), 0.5)
            PS_init = pt.zeros((self.n_subj, 4), dtype='float64')

            subj_idx_arr = pt.arange(self.n_subj)

            def step(c_t, x_t, mask_t, ug_prev, ul_prev, EF_prev, PS_prev, 
                     rho_, lam_, Arew_, Apun_, K_val_, beta_p_, theta_):
                
                EU_prev = EF_prev * ug_prev + (1.0 - EF_prev) * ul_prev
                V_prev = EU_prev + beta_p_[:, None] * PS_prev
                
                logits = V_prev * theta_[:, None]
                logits = logits - pt.max(logits, axis=1, keepdims=True)
                probs = pt.exp(logits) / pt.sum(pt.exp(logits), axis=1, keepdims=True)
                probs = pt.clip(probs, 1e-6, 1.0 - 1e-6)
                
                prob_chosen = probs[subj_idx_arr, c_t]
                ll_t = pt.log(prob_chosen) * mask_t
                
                is_pos = pt.ge(x_t, 0.0)
                curr_ug = pt.switch(is_pos, pt.power(pt.abs(x_t), rho_), ug_prev[subj_idx_arr, c_t])
                curr_ul = pt.switch(is_pos, ul_prev[subj_idx_arr, c_t], -lam_ * pt.power(pt.abs(x_t), rho_))
                
                ug_next = pt.set_subtensor(ug_prev[subj_idx_arr, c_t], curr_ug)
                ul_next = pt.set_subtensor(ul_prev[subj_idx_arr, c_t], curr_ul)
                
                curr_EF = pt.switch(is_pos, 
                                    EF_prev[subj_idx_arr, c_t] + Arew_ * (1.0 - EF_prev[subj_idx_arr, c_t]),
                                    EF_prev[subj_idx_arr, c_t] - Apun_ * EF_prev[subj_idx_arr, c_t])
                EF_next = pt.set_subtensor(EF_prev[subj_idx_arr, c_t], curr_EF)
                
                denom = 1.0 + K_val_
                PS_decayed = PS_prev / denom[:, None]
                PS_next = pt.set_subtensor(PS_decayed[subj_idx_arr, c_t], 1.0 / denom)
                
                return ug_next, ul_next, EF_next, PS_next, ll_t, probs

            [ug_scan, ul_scan, EF_scan, PS_scan, LLs, Probs], _ = scan(
                fn=step,
                sequences=[pt.as_tensor_variable(self.choices), 
                           pt.as_tensor_variable(self.outcomes),
                           pt.as_tensor_variable(self.mask)],
                outputs_info=[u_gain_init, u_loss_init, EF_init, PS_init, None, None],
                non_sequences=[rho, lam, Arew, Apun, K_val, beta_p, theta]
            )

            pm.Potential('likelihood', pt.sum(LLs))
            pm.Deterministic('Probs', Probs)
            
        return self.model

    def fit(self, draws=1000, tune=1000, chains=4):
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, 
                                   target_accept=0.95, return_inferencedata=True, 
                                   compute_convergence_checks=True)
        return self.trace
    
    def calculate_metrics(self):
        summary = az.summary(self.trace, var_names=['rho', 'lam', 'Arew', 'Apun', 'K_prime', 'beta_p', 'theta'])
        r_hat_max = summary['r_hat'].max()
        print(f"\n--- Диагностика MCMC ---")
        print(f"Максимальный Gelman-Rubin (R_hat): {r_hat_max:.4f} (в идеале < 1.05)")
        
    def plot_traces(self):
        az.plot_trace(self.trace, var_names=['mu_rho', 'mu_lam', 'mu_Arew', 'mu_Apun', 'mu_K_prime', 'mu_beta_p'])
        plt.tight_layout()
        plt.savefig("igt_perseveration_trace.png")
        print("Trace-график сохранен как 'igt_perseveration_trace.png'.")

    def simulate_subject(self, rho, lam, Arew, Apun, K_prime, beta_p, theta, n_trials=150):
        """Ручная симуляция агента с классической схемой выплат IGT."""
        K_val = (3.0 ** K_prime) - 1.0
        
        u_gain = np.zeros(4)
        u_loss = np.zeros(4)
        EF = np.full(4, 0.5)
        PS = np.zeros(4)
        
        choices = []
        outcomes = []
        
        for t in range(n_trials):
            EU = EF * u_gain + (1.0 - EF) * u_loss
            V = EU + beta_p * PS
            
            logits = V * theta
            logits = logits - np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            
            choice = np.random.choice(4, p=probs)
            choices.append(choice)
            
            # Генерация исхода по вашей схеме выплат IGT (чистый исход gain + loss)
            if choice == 0:    # Колода A: Награда 100, Штраф 250 (p=0.5)
                outcome_unscaled = -150 if np.random.rand() < 0.5 else 100
            elif choice == 1:  # Колода B: Награда 100, Штраф 625 (p=0.2)
                outcome_unscaled = -525 if np.random.rand() < 0.2 else 100
            elif choice == 2:  # Колода C: Награда 50, Штраф 50 (p=0.5)
                outcome_unscaled = 0 if np.random.rand() < 0.5 else 50
            else:              # Колода D (choice == 3): Награда 50, Штраф 125 (p=0.2)
                outcome_unscaled = -75 if np.random.rand() < 0.2 else 50
                
            outcomes.append(outcome_unscaled)
            outcome_scaled = outcome_unscaled / 100.0
            
            if outcome_scaled >= 0:
                u_gain[choice] = outcome_scaled ** rho
                EF[choice] = EF[choice] + Arew * (1.0 - EF[choice])
            else:
                u_loss[choice] = -lam * (abs(outcome_scaled) ** rho)
                EF[choice] = EF[choice] - Apun * EF[choice]
            
            PS = PS / (1.0 + K_val)
            PS[choice] += 1.0 / (1.0 + K_val)
            
        return pd.DataFrame({'trial': np.arange(n_trials), 'choice': choices, 'outcome': outcomes})

    def posterior_predictive_check(self, n_sims=100, save_path="igt_perseveration_ppc.png"):
        """PPC с сохранением метрик и графика."""
        print("[*] Запуск Posterior Predictive Check...")
        if self.trace is None:
            raise ValueError("Сначала обучите модель!")
            
        post = self.trace.posterior
        n_draws = post.dims['draw'] * post.dims['chain']
        sample_indices = np.random.choice(n_draws, size=n_sims, replace=False)
        
        real_adv_rate = []
        for s_idx in range(self.n_subj):
            subj_choices = self.choices[:, s_idx][self.mask[:, s_idx] == 1]
            real_adv_rate.append(np.mean(np.isin(subj_choices, [2, 3])))
        real_adv_mean = np.mean(real_adv_rate)
        
        sim_adv_means = []
        
        rho_post = post['rho'].values.reshape(-1, self.n_subj)
        lam_post = post['lam'].values.reshape(-1, self.n_subj)
        Arew_post = post['Arew'].values.reshape(-1, self.n_subj)
        Apun_post = post['Apun'].values.reshape(-1, self.n_subj)
        K_prime_post = post['K_prime'].values.reshape(-1, self.n_subj)
        beta_p_post = post['beta_p'].values.reshape(-1, self.n_subj)
        theta_post = post['theta'].values.reshape(-1, self.n_subj)
        
        for idx in tqdm(sample_indices, desc="Симуляция PPC"):
            sim_adv_s = []
            for s_idx in range(self.n_subj):
                sim_df = self.simulate_subject(
                    rho_post[idx, s_idx], lam_post[idx, s_idx], Arew_post[idx, s_idx], 
                    Apun_post[idx, s_idx], K_prime_post[idx, s_idx], beta_p_post[idx, s_idx], 
                    theta_post[idx, s_idx], n_trials=self.n_trials
                )
                adv = np.mean(sim_df['choice'].isin([2, 3]))
                sim_adv_s.append(adv)
            sim_adv_means.append(np.mean(sim_adv_s))
            
        ppp = np.mean(np.array(sim_adv_means) > real_adv_mean)
        
        # Сохранение графика
        plt.figure(figsize=(10, 6))
        sns.histplot(sim_adv_means, kde=True, color='skyblue', stat='density')
        plt.axvline(real_adv_mean, color='red', linestyle='--', label=f'Real Mean: {real_adv_mean:.3f}')
        plt.title(f'PPC: Advantageous Choices (Bayesian p-value = {ppp:.3f})')
        plt.xlabel('Proportion of Advantageous Choices')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"График PPC сохранен как '{save_path}'")
        
        # Возвращаем метрики как DataFrame для удобного сохранения
        metrics_df = pd.DataFrame([{
            'Real_Adv_Mean': real_adv_mean,
            'Sim_Adv_Mean': np.mean(sim_adv_means),
            'Bayesian_p_value_ppp': ppp
        }])
        
        return metrics_df

    def parameter_recovery(self, n_subjects=30, n_trials=100, save_path="igt_perseveration_recovery.png"):
        """Эмпирическое восстановление с сохранением графика Recovery."""
        print(f"[*] Запуск HBA Parameter Recovery (N={n_subjects})...")
        if self.trace is None:
            raise ValueError("Сначала обучите модель на реальных данных!")
            
        post = self.trace.posterior
        rng = np.random.default_rng(42)
        
        true_params = {
            'rho': rng.choice(post['rho'].mean(dim=["chain", "draw"]).values, n_subjects),
            'lam': rng.choice(post['lam'].mean(dim=["chain", "draw"]).values, n_subjects),
            'Arew': rng.choice(post['Arew'].mean(dim=["chain", "draw"]).values, n_subjects),
            'Apun': rng.choice(post['Apun'].mean(dim=["chain", "draw"]).values, n_subjects),
            'K_prime': rng.choice(post['K_prime'].mean(dim=["chain", "draw"]).values, n_subjects),
            'beta_p': rng.choice(post['beta_p'].mean(dim=["chain", "draw"]).values, n_subjects),
            'theta': rng.choice(post['theta'].mean(dim=["chain", "draw"]).values, n_subjects),
        }
        
        sim_data = []
        for i in range(n_subjects):
            df = self.simulate_subject(
                true_params['rho'][i], true_params['lam'][i], true_params['Arew'][i],
                true_params['Apun'][i], true_params['K_prime'][i], true_params['beta_p'][i],
                true_params['theta'][i], n_trials=n_trials
            )
            df['subj_id'] = f'sim_{i}'
            sim_data.append(df)
            
        sim_df = pd.concat(sim_data).reset_index(drop=True)
        
        rec_model = IGTPerseverationModel(sim_df)
        rec_model.build_model()
        rec_trace = rec_model.fit(draws=1000, tune=2000, chains=4)
        
        metrics = []
        for param in ['rho', 'lam', 'Arew', 'Apun', 'K_prime', 'beta_p', 'theta']:
            t_val = true_params[param]
            f_val = rec_trace.posterior[param].mean(dim=["chain", "draw"]).values
            hdi = az.hdi(rec_trace, var_names=[param], hdi_prob=0.95)[param].values
            
            r = np.corrcoef(t_val, f_val)[0, 1]
            rmse = np.sqrt(mean_squared_error(t_val, f_val))
            bias = np.mean(f_val - t_val)
            coverage = np.mean((t_val >= hdi[:, 0]) & (t_val <= hdi[:, 1]))
            
            metrics.append({'Param': param, 'r': r, 'RMSE': rmse, 'Bias': bias, 'Coverage': coverage})
            
        metrics_df = pd.DataFrame(metrics)
        print("\nМетрики Parameter Recovery:")
        print(metrics_df)

        # Создаем датафрейм, объединяющий истинные и восстановленные параметры
        params_dict_for_df = {'subj_id': [f'sim_{i}' for i in range(n_subjects)]}
        for param in ['rho', 'lam', 'Arew', 'Apun', 'K_prime', 'beta_p', 'theta']:
            params_dict_for_df[f'true_{param}'] = true_params[param]
            params_dict_for_df[f'rec_{param}'] = rec_trace.posterior[param].mean(dim=["chain", "draw"]).values
            
        params_df = pd.DataFrame(params_dict_for_df)
        
        # Сохранение графика (Scatter Plots)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        for idx, param in enumerate(['rho', 'lam', 'Arew', 'Apun', 'K_prime', 'beta_p', 'theta']):
            t_val = true_params[param]
            f_val = rec_trace.posterior[param].mean(dim=["chain", "draw"]).values
            r_val = metrics_df.loc[metrics_df['Param'] == param, 'r'].values[0]
            
            sns.scatterplot(x=t_val, y=f_val, ax=axes[idx], color='coral')
            axes[idx].plot([min(t_val), max(t_val)], [min(t_val), max(t_val)], 'k--')
            axes[idx].set_title(f"{param} (r={r_val:.2f})")
            axes[idx].set_xlabel('True values')
            axes[idx].set_ylabel('Recovered values')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Графики Recovery сохранены как '{save_path}'")
        
        # Возвращаем params_df вместо словаря true_params
        return params_df, rec_trace, metrics_df