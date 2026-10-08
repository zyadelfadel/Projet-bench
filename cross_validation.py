import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Charger la série globale
weekly = pd.read_csv("bench_weekly.csv")
weekly["ds"] = pd.to_datetime(weekly["ds"])

# ---- Modèle 1 : baseline ----
print("=== Cross-validation : modèle baseline ===")
model_baseline = Prophet()
model_baseline.fit(weekly)

df_cv_baseline = cross_validation(
    model_baseline,
    initial="365 days",
    period="90 days",
    horizon="90 days",
)
metrics_baseline = performance_metrics(df_cv_baseline)
print(metrics_baseline[["horizon", "mae", "mape"]])
mae_moyen_baseline = metrics_baseline["mae"].mean()

# ---- Modèle 2 : avec holidays ----
print("\n=== Cross-validation : modèle avec holidays ===")
ete = pd.DataFrame({
    "holiday": "vacances_ete",
    "ds": pd.to_datetime(["2023-08-01", "2024-08-01", "2025-08-01"]),
    "lower_window": -15,
    "upper_window": 15,
})
fin_annee = pd.DataFrame({
    "holiday": "fetes_fin_annee",
    "ds": pd.to_datetime(["2023-12-25", "2024-12-25", "2025-12-25"]),
    "lower_window": -10,
    "upper_window": 10,
})
holidays = pd.concat([ete, fin_annee])

model_holidays = Prophet(holidays=holidays)
model_holidays.fit(weekly)

df_cv_holidays = cross_validation(
    model_holidays,
    initial="365 days",
    period="90 days",
    horizon="90 days",
)
metrics_holidays = performance_metrics(df_cv_holidays)
print(metrics_holidays[["horizon", "mae", "mape"]])
mae_moyen_holidays = metrics_holidays["mae"].mean()

# ---- Comparaison finale ----
print("\n=== Comparaison finale (moyenne sur toutes les fenêtres de test) ===")
print(f"Modèle baseline      : MAE moyen = {mae_moyen_baseline:.2f}")
print(f"Modèle avec holidays : MAE moyen = {mae_moyen_holidays:.2f}")

if mae_moyen_holidays < mae_moyen_baseline:
    print("\n-> Sur plusieurs fenêtres de test, le modèle avec holidays est globalement plus précis.")
else:
    print("\n-> Sur plusieurs fenêtres de test, le modèle baseline reste aussi bon, voire meilleur.")