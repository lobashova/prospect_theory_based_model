import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from models.hcct_v2_model import HCCTModel_v2
import os

def main():
    print("=== ЗАПУСК ПОДГОНКИ МОДЕЛИ hCCT (Итоговая модель v2) ===")
    
    # 1. Загрузка данных (замените на ваш файл, если нужно)
    # Ожидается формат: user_id, trial_number, choice, flips, popped, gain_amount, loss_amount, loss_cards
    df = pd.read_csv('data/hot_data.csv') 
    
    # 2. Инициализация и запуск HBA
    model = HCCTModel_v2(n_cards=32)
    trace, users = model.fit(df, draws=1500, tune=1000, chains=4, cores=4)
    az.to_netcdf(trace, "hcct_v2_trace.nc")

    # 3. Сохранение графиков Trace (чтобы убедиться в сходимости)
    print("\nСохранение графиков Trace...")
    az.plot_trace(trace, var_names=['mu_rho', 'mu_lam', 'mu_theta', 'mu_Arew', 'mu_Apun', 'mu_gamma'])
    plt.tight_layout()
    plt.savefig('hcct_v2_trace.png', dpi=300)
    plt.close()
    
    # 4. Извлечение и сохранение параметров
    summary = az.summary(trace, var_names=['rho', 'lam', 'theta', 'Arew', 'Apun', 'gamma'])
    summary.to_csv('hcct_v2_hba_summary.csv')
    
    # 5. Выполнение PPC (R^2, RMSE, MAE, Hit Rate, ppp)
    ppc_metrics = model.posterior_predictive_check(trace, df)
    ppc_metrics.to_csv('hcct_v2_ppc_metrics.csv', index=False)

    # 6. Запуск True HBA Parameter Recovery
    print("\n[*] Запуск Parameter Recovery для hCCT...")
    recovery_df, rec_metrics = model.parameter_recovery(trace, df, n_subjects=50)
    
    # Сохраняем индивидуальные параметры и общие метрики[cite: 44]
    if recovery_df is not None and rec_metrics is not None:
        recovery_df.to_csv('hcct_v2_recovery_data.csv', index=False)
        rec_metrics.to_csv('hcct_v2_recovery_metrics.csv', index=False)
        print("[✓] Результаты Parameter Recovery успешно сохранены.")
    
    print("\n=== ВСЕ РАСЧЕТЫ ДЛЯ hCCT ЗАВЕРШЕНЫ УСПЕШНО ===")

if __name__ == '__main__':
    main()