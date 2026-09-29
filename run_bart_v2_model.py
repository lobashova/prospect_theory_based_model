import os
import pandas as pd
import numpy as np
import arviz as az
from models.bart_v2_model import BART_V2_Model_HBA

def main():
    print("=== ИТОГОВАЯ МОДЕЛЬ V2 (BART) ===")
    # os.makedirs("results", exist_ok=True)
    
    # 1. Загрузка данных (замените на чтение вашего CSV)
    data_path = 'data/bart_data.csv'
    df = pd.read_csv(data_path)

    # 2. Инициализация и подгонка
    model = BART_V2_Model_HBA(df, scale_factor=10.0)
    model.fit(draws=1500, tune=1500, chains=4, cores=4)
    
    # Сохранение Trace (опционально, для последующего анализа)
    az.to_netcdf(model.trace, "results/bart_v2_trace.nc")
    
    # Вывод метрик сходимости (R-hat)
    summary_df = az.summary(model.trace, var_names=['rho', 'lam', 'phi', 'Arew', 'Apun', 'eps', 'theta'])
    summary_df.to_csv("results/bart_v2_summary.csv")
    print("\n--- Проверка сходимости цепей (R-hat) ---")
    print(summary_df[['mean', 'r_hat']])
    
    # 3. Posterior Predictive Check
    ppc_metrics = model.posterior_predictive_check(n_sims=100, save_path="results")
    ppc_metrics.to_csv("results/bart_v2_ppc_metrics.csv", index=False)
    print("\n--- Метрики PPC ---")
    print(ppc_metrics)
    
    # 4. Parameter Recovery (с сохранением всех истинных/симулированных параметров)
    rec_data, rec_metrics = model.parameter_recovery(n_subjects=20, n_trials=40, save_path="results")
    rec_data.to_csv("results/bart_v2_recovery_params.csv", index=False)
    rec_metrics.to_csv("results/bart_v2_recovery_metrics.csv", index=False)
    print("\n--- Метрики Parameter Recovery ---")
    print(rec_metrics)

if __name__ == "__main__":
    main()