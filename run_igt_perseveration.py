import os
import pandas as pd
import numpy as np
import arviz as az
import matplotlib

# Отключаем интерактивный бэкенд для безопасной работы на суперкомпьютере
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Импорт вашего обновленного класса модели
from models.igt_perseveration_model import IGTPerseverationModel

def main():
    print("=" * 70)
    print("=== ЗАПУСК IGT С ПЕРСЕВЕРАЦИЕЙ (HBA) ===")
    print("=" * 70)
    
    # 1. Загрузка данных
    print("[*] Генерация/загрузка данных...")
    df = pd.read_csv('data/igt_data.csv')
    
    # 2. Инициализация и компиляция модели
    print("[*] Инициализация модели IGT...")
    model = IGTPerseverationModel(df)
    model.build_model()
    trace = az.from_netcdf("igt_perseveration_trace.nc")
    model.trace = trace
    # # 3. Подгонка параметров (HBA MCMC)
    # print("[*] Запуск сэмплирования NUTS...")
    # # Для реального запуска увеличьте draws и tune до 1000/2000
    # trace = model.fit(draws=1000, tune=2000, chains=4)
    
    # # 4. Сохранение результатов подгонки
    # az.to_netcdf(trace, "igt_perseveration_trace.nc")
    # print("[✓] Trace модели сохранен в 'igt_perseveration_trace.nc'")

    # # 4.1 Сохранение подогнанных параметров в CSV
    # print("[*] Сохранение параметров в CSV...")
    # summary_df = az.summary(trace, var_names=['rho', 'lam', 'Arew', 'Apun', 'K_prime', 'beta_p', 'theta'])
    # summary_df.to_csv("igt_perseveration_hba_params.csv")
    # print("[✓] Параметры сохранены в 'igt_perseveration_hba_params.csv'")
    
    # # 5. Базовые метрики и графики
    # print("\n[*] Расчет базовых метрик MCMC...")
    # model.calculate_metrics()
    # model.plot_traces()
    
    # # 6. Posterior Predictive Check (PPC)
    # # Метод теперь сам строит график и возвращает DataFrame с метриками (ppp, mean)
    # print("\n[*] Запуск Posterior Predictive Check...")
    # ppc_metrics_df = model.posterior_predictive_check(n_sims=100, save_path="igt_perseveration_ppc.png")
    # ppc_metrics_df.to_csv("igt_perseveration_ppc_metrics.csv", index=False)
    # print("[✓] Метрики PPC сохранены в 'igt_perseveration_ppc_metrics.csv'")
    
    # 7. Parameter Recovery
    # Метод теперь сам строит графики восстановления (scatter plots)
    print("\n[*] Запуск Parameter Recovery...")
    true_params, rec_trace, metrics_df = model.parameter_recovery(
        n_subjects=20, 
        n_trials=150, 
        save_path="igt_perseveration_recovery_scatter.png"
    )
    
    # Сохранение метрик Recovery
    metrics_df.to_csv("igt_perseveration_recovery_metrics.csv", index=False)
    true_params.to_csv("igt_perseveration_recovery_params.csv", index=False)
    print("[✓] Метрики Recovery сохранены в 'igt_perseveration_recovery_metrics.csv'")
    
    print("\n" + "=" * 70)
    print("=== ВСЕ РАСЧЕТЫ ДЛЯ IGT С ПЕРСЕВЕРАЦИЕЙ ЗАВЕРШЕНЫ УСПЕШНО ===")
    print("=" * 70)

# Обязательная защита для корректной работы мультипроцессинга PyMC
if __name__ == '__main__':
    main()