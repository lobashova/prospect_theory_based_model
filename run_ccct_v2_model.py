import pandas as pd
import numpy as np
import arviz as az
import os
import matplotlib
matplotlib.use('Agg') # Безопасно для кластера без GUI
from models.ccct_v2_model import CCCTModel_v2

def main():
    print("=" * 70)
    print("=== ЗАПУСК ПОДГОНКИ И ПРОВЕРОК МОДЕЛИ cCCT (Итоговая модель v2) ===")
    print("=" * 70)

    def preprocess_ccct_data(df, n_cards=33):
        df_prep = df.copy()
        
        # 1. Переименование базовых колонок
        df_prep = df_prep.rename(columns={
            'num_cards': 'choice',
            'loss_encountered': 'is_penalty'
        })
        
        # В cCCT ЛПР выбирает число карт заранее, поэтому choice = k_cards_drawn
        df_prep['k_cards_drawn'] = df_prep['choice']
        df_prep['is_penalty'] = df_prep['is_penalty'].astype(int)
        
        # 2. Генерация матриц наград и объективных вероятностей для всех N (от 0 до 32)
        # Используем линейную аппроксимацию объективной вероятности из ваших прошлых скриптов
        for i in range(n_cards):
            df_prep[f'gain_n_{i}'] = i * 10  # Из примера: 15 карт = 150 очков, шаг = 10
            df_prep[f'loss_n_{i}'] = df_prep['loss_amount']
            # Вероятность успеха линейно убывает: 1 - N * (loss_cards / 32)
            p_obj = 1.0 - (i * df_prep['loss_cards'] / 32.0)
            df_prep[f'p_obj_n_{i}'] = np.clip(p_obj, 1e-6, 1.0)
            
        return df_prep

    # 1. Загрузка данных (замените путь на актуальный)
    df_cold = pd.read_csv('data/cold_data.csv')
    df_ready = preprocess_ccct_data(df_cold)
    # 2. Инициализация и подгонка (HBA, NUTS)
    model = CCCTModel_v2(n_cards=33, scale_factor=100.0) # scale_factor для CCT 100.0
    # trace, user_ids = model.fit(df_ready, draws=1500, tune=1000, chains=4, cores=4)
    
    # # 3. Сохранение Trace (опционально)
    # az.to_netcdf(trace, "ccct_v2_trace.nc")
    # print("\n[✓] Trace сохранен в ccct_v2_trace.nc")
    trace = az.from_netcdf("ccct_v2_trace.nc")
    model.trace = trace
    model.users = df_ready['user_id'].astype('category')
    model.subj_labels = model.users.cat.categories.values
    model.n_subj = len(model.subj_labels)
    
    # # 4. Проверка сходимости и сохранение сводки параметров (R_hat / Gelman-Rubin)
    # summary_df = az.summary(trace, var_names=['rho', 'lam', 'theta', 'Arew', 'Apun', 'gamma'])
    # summary_df.to_csv("ccct_v2_hba_params.csv")
    # print("\n=== Показатель Гельмана-Рубина (R-hat) ===")
    # print(summary_df[['mean', 'r_hat']].head(10))
    
    # if (summary_df['r_hat'] > 1.05).any():
    #     print("[!] Внимание: Есть параметры с R_hat > 1.05. Цепи могли не сойтись.")
    # else:
    #     print("[✓] Отличная сходимость: все R_hat <= 1.05")
    
    # # 5. Posterior Predictive Check
    # ppc_metrics = model.posterior_predictive_check(trace, df_ready, save_path="ccct_v2_ppc_timecourse.png")
    # ppc_metrics.to_csv("ccct_v2_ppc_metrics.csv", index=False)
    # print("\n=== PPC Metrics ===")
    # print(ppc_metrics)
    
    # 6. Parameter Recovery
    rec_data, rec_metrics = model.parameter_recovery(df_ready, n_subjects=20, n_trials=96)
    rec_data.to_csv("ccct_v2_recovery_data.csv", index=False)
    rec_metrics.to_csv("ccct_v2_recovery_metrics.csv", index=False)
    print("\n=== Recovery Metrics ===")
    print(rec_metrics)

if __name__ == '__main__':
    main()