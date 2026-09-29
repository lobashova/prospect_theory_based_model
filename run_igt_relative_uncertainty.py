import os
import pandas as pd
import numpy as np
import arviz as az
import matplotlib
# Отключаем интерактивный бэкенд для безопасной работы на суперкомпьютере[cite: 23]
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from models.igt_relative_uncertainty_model import IGTRelativeUncertaintyModel

def main():
    print("=" * 70)
    print("=== ЗАПУСК IGT С ОТНОСИТЕЛЬНОЙ НЕОПРЕДЕЛЕННОСТЬЮ (HBA) ===")
    print("=" * 70)
    
    # 1. Загрузка данных (замените на чтение вашего CSV)
    print("[*] Генерация/загрузка данных...")
    df = pd.read_csv('data/igt_data.csv')
    
    # 2. Инициализация
    print("[*] Инициализация модели IGT...")
    model = IGTRelativeUncertaintyModel(df, n_trials=150, scale_factor=100.0)
    
    # # 3. Подгонка (NUTS)
    # print("[*] Запуск сэмплирования NUTS...")
    # # Для реального запуска увеличьте draws и tune до 1500
    # trace, user_ids = model.fit(draws=1500, tune=1500, chains=4, cores=4)
    
    # Сохранение Trace (обязательно)
    # az.to_netcdf(trace, "igt_rel_unc_trace.nc")
    # print("[✓] Trace модели сохранен в 'igt_rel_unc_trace.nc'")
    trace = az.from_netcdf("igt_rel_unc_trace.nc")
    model.trace = trace
    
    # # 4. Проверка Гельмана-Рубина (R_hat) и сохранение параметров
    # summary_df = az.summary(trace, var_names=['rho', 'lam', 'Arew', 'Apun', 'a_unc', 'w', 'theta'])
    # summary_df.to_csv("igt_rel_unc_hba_params.csv")
    # print("[✓] Параметры сохранены в 'igt_rel_unc_hba_params.csv'")
    
    # print("\n=== Показатель Гельмана-Рубина (R-hat) ===")
    # print(summary_df[['mean', 'r_hat']].head(10))
    # if (summary_df['r_hat'] > 1.05).any():
    #     print("[!] Внимание: Есть параметры с R_hat > 1.05. Цепи могли не сойтись!")
    # else:
    #     print("[✓] Отличная сходимость: все R_hat <= 1.05")
    
    # # 5. Posterior Predictive Check
    # ppc_metrics = model.posterior_predictive_check(trace, df, block_size=20)
    # ppc_metrics.to_csv("igt_rel_unc_ppc_metrics.csv", index=False)
    # print("\n=== PPC Metrics ===")
    # print(ppc_metrics)
    
    # 6. Parameter Recovery
    rec_df, rec_metrics = model.parameter_recovery(n_subjects=20, n_trials=150)
    # rec_metrics.to_csv("igt_rel_unc_recovery_metrics.csv", index=False)
    print("\n=== Recovery Metrics ===")
    print(rec_metrics)
    
    print("\n" + "=" * 70)
    print("=== ВСЕ РАСЧЕТЫ ЗАВЕРШЕНЫ УСПЕШНО ===")
    print("=" * 70)

# Обязательная защита для корректной работы мультипроцессинга PyMC[cite: 18, 34, 46]
if __name__ == '__main__':
    main()