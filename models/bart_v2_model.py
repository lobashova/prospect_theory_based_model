import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # HPC-safe режим
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az

warnings.filterwarnings("ignore")

def concordance_correlation_coefficient(y_true, y_pred):
    """Расчет CCC (Concordance Correlation Coefficient)"""
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator

class BART_V2_Model_HBA:
    def __init__(self, data_df, scale_factor=10.0):
        self.data = data_df
        self.scale_factor = scale_factor
        self.r = 1.0 / self.scale_factor 
        self.max_pumps = 64
        self.trace = None
        
        if data_df is not None:
            self._prepare_data()

    def _prepare_data(self):
        self.subjects = self.data['user_id'].unique()
        self.n_subj = len(self.subjects)
        self.max_trials = self.data.groupby('user_id').size().max()
        
        self.pumps_mat = np.zeros((self.n_subj, self.max_trials), dtype=int)
        self.popped_mat = np.zeros((self.n_subj, self.max_trials), dtype=int)
        self.valid_mat = np.zeros((self.n_subj, self.max_trials), dtype=int)
        self.n_trials_arr = np.zeros(self.n_subj, dtype=int)
        
        for i, subj in enumerate(self.subjects):
            subj_data = self.data[self.data['user_id'] == subj]
            n_t = len(subj_data)
            self.n_trials_arr[i] = n_t
            self.pumps_mat[i, :n_t] = subj_data['pumps'].values
            self.popped_mat[i, :n_t] = subj_data['popped'].values
            self.valid_mat[i, :n_t] = 1

    def fit(self, draws=1500, tune=1500, chains=4, cores=4):
        print(f"[*] Запуск HBA (NUTS) для {self.n_subj} участников...")
        
        coords = {"subj": self.subjects}
        # ИЗМЕНЕНО: Передаем координаты в модель
        with pm.Model(coords=coords) as self.model:
            # Гиперпараметры (NCP)
            mu_rho = pm.Normal('mu_rho', 0, 1)
            sigma_rho = pm.HalfNormal('sigma_rho', 1)
            mu_lam = pm.Normal('mu_lam', 0, 1)
            sigma_lam = pm.HalfNormal('sigma_lam', 1)
            mu_phi = pm.Normal('mu_phi', 0, 1)
            sigma_phi = pm.HalfNormal('sigma_phi', 1)
            mu_Arew = pm.Normal('mu_Arew', 0, 1)
            sigma_Arew = pm.HalfNormal('sigma_Arew', 1)
            mu_Apun = pm.Normal('mu_Apun', 0, 1)
            sigma_Apun = pm.HalfNormal('sigma_Apun', 1)
            mu_eps = pm.Normal('mu_eps', 0, 1)
            sigma_eps = pm.HalfNormal('sigma_eps', 1)
            mu_theta = pm.Normal('mu_theta', 0, 1)
            sigma_theta = pm.HalfNormal('sigma_theta', 1)
            
            # Индивидуальные отклонения
            z_rho = pm.Normal('z_rho', 0, 1, dims="subj")
            z_lam = pm.Normal('z_lam', 0, 1, dims="subj")
            z_phi = pm.Normal('z_phi', 0, 1, dims="subj")
            z_Arew = pm.Normal('z_Arew', 0, 1, dims="subj")
            z_Apun = pm.Normal('z_Apun', 0, 1, dims="subj")
            z_eps = pm.Normal('z_eps', 0, 1, dims="subj")
            z_theta = pm.Normal('z_theta', 0, 1, dims="subj")
            
            # Трансформации
            rho = pm.Deterministic('rho', 2.0 * pm.math.invlogit(mu_rho + z_rho * sigma_rho)) 
            lam = pm.Deterministic('lam', 10.0 * pm.math.invlogit(mu_lam + z_lam * sigma_lam))
            phi = pm.Deterministic('phi', pm.math.invlogit(mu_phi + z_phi * sigma_phi))
            Arew = pm.Deterministic('Arew', pm.math.invlogit(mu_Arew + z_Arew * sigma_Arew)) 
            Apun = pm.Deterministic('Apun', pm.math.invlogit(mu_Apun + z_Apun * sigma_Apun)) 
            eps = pm.Deterministic('eps', 5.0 * pm.math.invlogit(mu_eps + z_eps * sigma_eps))
            theta = pm.Deterministic('theta', 15.0 * pm.math.invlogit(mu_theta + z_theta * sigma_theta))

            N_mat = pt.constant(np.arange(1, self.max_pumps + 1, dtype='float64')[None, :])
            P_init = phi[:, None] ** N_mat
            
            def step_fn(K_t, popped_t, valid_t, P_prev, Arew, Apun, eps, rho, lam, theta, r):
                K_mat = K_t[:, None]
                u_gain = (N_mat * r) ** rho[:, None]
                u_loss = -lam[:, None] * ((N_mat - 1) * r) ** rho[:, None]
                
                EU = P_prev * u_gain + (1.0 - P_prev) * u_loss
                logits = EU * theta[:, None]
                logits_shifted = logits - pt.max(logits, axis=1, keepdims=True)
                exp_logits = pt.exp(logits_shifted)
                probs = exp_logits / pt.sum(exp_logits, axis=1, keepdims=True)
                
                idx = pt.cast(pt.clip(K_t - 1, 0, 63), 'int32')
                p_choice = probs[pt.arange(K_t.shape[0]), idx]
                logp_t = pt.log(pt.clip(p_choice, 1e-12, 1.0)) * valid_t
                
                mask_success = pt.eq(popped_t, 0)[:, None]
                mask_popped = pt.eq(popped_t, 1)[:, None]
                
                mask_le_K = pt.le(N_mat, K_mat)
                mask_gt_K = pt.gt(N_mat, K_mat)
                mask_ge_K = pt.ge(N_mat, K_mat)
                mask_lt_K = pt.lt(N_mat, K_mat)
                
                G = pt.exp(-eps[:, None] * (N_mat - K_mat))
                
                update_succ = mask_success * (mask_le_K * Arew[:, None] * (1.0 - P_prev) + 
                                              mask_gt_K * Arew[:, None] * G * (1.0 - P_prev))
                update_pop = mask_popped * (mask_ge_K * -Apun[:, None] * P_prev + 
                                            mask_lt_K * Arew[:, None] * (1.0 - P_prev))
                
                P_new = P_prev + valid_t[:, None] * (update_succ + update_pop)
                return P_new, logp_t

            [_, logp_seq], _ = scan(
                fn=step_fn,
                sequences=[pt.as_tensor_variable(self.pumps_mat.T), 
                           pt.as_tensor_variable(self.popped_mat.T), 
                           pt.as_tensor_variable(self.valid_mat.T)],
                outputs_info=[P_init, None],
                non_sequences=[Arew, Apun, eps, rho, lam, theta, self.r]
            )
            
            user_logp = pt.sum(logp_seq, axis=0)
            pm.Deterministic('log_likelihood', user_logp)
            pm.Potential('likelihood_sum', pt.sum(user_logp))
            
            # По умолчанию PyMC использует NUTS (без необходимости сторонних библиотек вроде nutpie)
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, 
                                   target_accept=0.9, return_inferencedata=True, progressbar=False)
            
            return self.trace

    def simulate_subject(self, params, n_trials=30, empirical_break_points=None):
        rho, lam, phi, Arew, Apun, eps, theta = params
        N = np.arange(1, self.max_pumps + 1)
        P = np.power(phi, N)
        u_gain = np.power(N * self.r, rho)
        u_loss = -lam * np.power((N - 1) * self.r, rho)
        
        sims = []
        for t in range(n_trials):
            EU = P * u_gain + (1.0 - P) * u_loss
            logits = EU * theta
            logits -= np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            
            K = np.random.choice(N, p=probs)
            bp = empirical_break_points[t] if empirical_break_points is not None else np.random.randint(1, 65)
            popped = 1 if K >= bp else 0
            actual_K = bp if popped else K
            
            sims.append({'trial': t+1, 'pumps': actual_K, 'popped': popped, 'break_point': bp})
            
            new_P = P.copy()
            if not popped:
                new_P[N <= actual_K] += Arew * (1.0 - P[N <= actual_K])
                new_P[N > actual_K] += Arew * np.exp(-eps * (N[N > actual_K] - actual_K)) * (1.0 - P[N > actual_K])
            else:
                new_P[N >= actual_K] -= Apun * P[N >= actual_K]
                new_P[N < actual_K] += Arew * (1.0 - P[N < actual_K])
            P = new_P
            
        return pd.DataFrame(sims)

    def posterior_predictive_check(self, n_sims=100, save_path="results"):
        print("[*] Выполнение Posterior Predictive Check...")
        # os.makedirs(save_path, exist_ok=True)
        
        post = self.trace.posterior
        n_chains, n_draws = post['rho'].shape[0], post['rho'].shape[1]
        total_samples = n_chains * n_draws
        
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(total_samples, size=n_sims, replace=False)
        
        all_real, all_sim = [], []
        hit_rates, msd_vals, ppp_vals = [], [], []
        
        for i, subj in enumerate(self.subjects):
            subj_data = self.data[self.data['user_id'] == subj]
            n_t = len(subj_data)
            bps = subj_data['break_point'].values if 'break_point' in subj_data.columns else None
            real_pumps = subj_data['pumps'].values
            
            subj_sims = np.zeros((n_sims, n_t))
            adj_scores = []
            
            for sim_idx, flat_idx in enumerate(sample_indices):
                c, d = flat_idx // n_draws, flat_idx % n_draws
                params = (post[p][c, d, i].values for p in ['rho', 'lam', 'phi', 'Arew', 'Apun', 'eps', 'theta'])
                sim_df = self.simulate_subject(tuple(params), n_t, bps)
                subj_sims[sim_idx, :] = sim_df['pumps'].values
                adj_scores.append(sim_df[sim_df['popped']==0]['pumps'].mean())
            
            sim_mean = subj_sims.mean(axis=0)
            mode_sim = pd.DataFrame(subj_sims).mode(axis=0).iloc[0].values
            
            real_adj = subj_data[subj_data['popped']==0]['pumps'].mean()
            
            all_real.extend(real_pumps)
            all_sim.extend(sim_mean)
            hit_rates.append(np.mean(real_pumps == mode_sim))
            msd_vals.append(np.mean((real_pumps - sim_mean)**2))
            ppp_vals.append(np.mean(np.array(adj_scores) > real_adj))

        metrics = pd.DataFrame([{
            'R2': r2_score(all_real, all_sim), 
            'RMSE': np.sqrt(mean_squared_error(all_real, all_sim)), 
            'MAE': mean_absolute_error(all_real, all_sim), 
            'Hit_Rate': np.mean(hit_rates), 
            'MSD': np.mean(msd_vals), 
            'PPP': np.mean(ppp_vals)
        }])
        return metrics

    def parameter_recovery(self, n_subjects=30, n_trials=40, draws=1000, tune=1000, save_path="results"):
        print(f"[*] Запуск Parameter Recovery ({n_subjects} субъектов)...")
        # os.makedirs(save_path, exist_ok=True)
        
        post = az.summary(self.trace, var_names=['rho', 'lam', 'phi', 'Arew', 'Apun', 'eps', 'theta'])
        
        def get_t(name, a, b):
            m, s = post.loc[post.index.str.startswith(name), 'mean'].mean(), post.loc[post.index.str.startswith(name), 'sd'].mean()
            return truncnorm.rvs((a - m) / s, (b - m) / s, loc=m, scale=s, size=n_subjects)

        true_p = {
            'rho': get_t('rho', 0.01, 2.0), 'lam': get_t('lam', 0.01, 10.0), 'phi': get_t('phi', 0.01, 0.99),
            'Arew': get_t('Arew', 0.01, 0.99), 'Apun': get_t('Apun', 0.01, 0.99), 'eps': get_t('eps', 0.01, 5.0),
            'theta': get_t('theta', 0.01, 15.0)
        }
        
        sim_data = []
        for i in range(n_subjects):
            p_tuple = (true_p['rho'][i], true_p['lam'][i], true_p['phi'][i], true_p['Arew'][i], true_p['Apun'][i], true_p['eps'][i], true_p['theta'][i])
            df_s = self.simulate_subject(p_tuple, n_trials)
            df_s['user_id'] = f"sim_{i}"
            sim_data.append(df_s)
            
        rec_model = BART_V2_Model_HBA(pd.concat(sim_data, ignore_index=True), self.scale_factor)
        rec_model.fit(draws=draws, tune=tune, chains=4, cores=4)
        
        rec_post = rec_model.trace.posterior
        metrics_list = []
        params_df = pd.DataFrame({'subj_id': [f"sim_{i}" for i in range(n_subjects)]})
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, p_name in enumerate(true_p.keys()):
            true_vals = true_p[p_name]
            fit_vals = rec_post[p_name].mean(dim=["chain", "draw"]).values
            hdi = az.hdi(rec_model.trace, var_names=[p_name], hdi_prob=0.95)[p_name].values
            
            r = pearsonr(true_vals, fit_vals)[0]
            r2 = r2_score(true_vals, fit_vals)
            rmse = np.sqrt(mean_squared_error(true_vals, fit_vals))
            bias = np.mean(fit_vals - true_vals)
            coverage = np.mean((true_vals >= hdi[:, 0]) & (true_vals <= hdi[:, 1]))
            ccc = concordance_correlation_coefficient(true_vals, fit_vals)
            
            params_df[f'true_{p_name}'] = true_vals
            params_df[f'fit_{p_name}'] = fit_vals
            metrics_list.append({'Param': p_name, 'r': r, 'R2': r2, 'RMSE': rmse, 'Bias': bias, 'Coverage': coverage, 'CCC': ccc})
            
            ax = axes[idx]
            ax.errorbar(true_vals, fit_vals, yerr=[fit_vals - hdi[:, 0], hdi[:, 1] - fit_vals], fmt='o', alpha=0.6)
            vmin, vmax = min(true_vals), max(true_vals)
            ax.plot([vmin, vmax], [vmin, vmax], 'k--')
            ax.set_title(f"{p_name}\nr={r:.2f}, Cov={coverage*100:.0f}%, CCC={ccc:.2f}")
            
        plt.tight_layout()
        plt.savefig(f"recovery_plot.png", dpi=300)
        plt.close(fig)
        
        return params_df, pd.DataFrame(metrics_list)
