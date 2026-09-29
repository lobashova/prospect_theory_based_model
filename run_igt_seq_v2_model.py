import os
import pandas as pd
import numpy as np
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Выключаем графический бэкенд для кластера
from models.igt_seq_v2_model import IGTSequentialV2Model

def main():
    print("=" * 70)
    print("=== IGT SEQUENTIAL V2 (HBA, NUTS, 150 trials) ===")
    print("=" * 70)
    
    # 1. Загрузка данных
    df = pd.read_csv('data/igt_data.csv')
    
    # 2. Инициализация (150 триалов)
    model = IGTSequentialV2Model(df, n_trials=150)
    trace = az.from_netcdf("igt_seq_v2_trace.nc")
    model.trace = trace
    # 3. Подгонка (NUTS)
    # trace, user_ids = model.fit(draws=1500, tune=1500, chains=4, cores=4)
    
    # # Сохранение Trace (обязательно)
    # az.to_netcdf(trace, "igt_seq_v2_trace.nc")
    
    # # 4. Проверка Гельмана-Рубина (R_hat)
    # summary_df = az.summary(trace, var_names=['rho', 'lam', 'Arew', 'Apun', 'alpha', 'phi', 'theta'])
    # summary_df.to_csv("igt_seq_v2_hba_params.csv")
    
    # print("\n=== Показатель Гельмана-Рубина (R-hat) ===")
    # print(summary_df[['mean', 'r_hat']].head(10))
    # if (summary_df['r_hat'] > 1.05).any():
    #     print("[!] Внимание: Есть параметры с R_hat > 1.05. Цепи могли не сойтись!")
    # else:
    #     print("[✓] Отличная сходимость: все R_hat <= 1.05")
    
    # 5. Posterior Predictive Check (разбиваем 150 попыток на 5 блоков по 30)
    ppc_metrics = model.posterior_predictive_check(trace, df, block_size=30)
    ppc_metrics.to_csv("igt_seq_v2_ppc_metrics.csv", index=False)
    print("\n=== PPC Metrics ===")
    print(ppc_metrics)
    
    # # 6. Parameter Recovery
    # rec_df, rec_metrics = model.parameter_recovery(n_subjects=20, n_trials=150)
    # # rec_metrics.to_csv("igt_seq_v2_recovery_metrics.csv", index=False)
    # print("\n=== Recovery Metrics ===")
    # print(rec_metrics)

if __name__ == '__main__':
    main()