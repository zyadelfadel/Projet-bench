import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Charger la série globale
weekly = pd.read_csv("bench_weekly.csv")
weekly["ds"] = pd.to_datetime(weekly["ds"])

# Split train/test : on garde les 12 dernières semaines comme test
n_test = 12
train = weekly.iloc[:-n_test]
test = weekly.iloc[-n_test:]

print(f"Train : {len(train)} semaines, Test : {len(test)} semaines")

# ---- Modèle 1 : baseline ----
model_baseline = Prophet()
model_baseline.fit(train)

future_baseline = model_baseline.make_future_dataframe(periods=n_test, freq="W")
forecast_baseline = model_baseline.predict(future_baseline)

pred_baseline = forecast_baseline.set_index("ds").loc[test["ds"], "yhat"]
mae_baseline = mean_absolute_error(test["y"], pred_baseline)
mape_baseline = mean_absolute_percentage_error(test["y"], pred_baseline) * 100

# ---- Modèle 2 : avec holidays ----
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
model_holidays.fit(train)

future_holidays = model_holidays.make_future_dataframe(periods=n_test, freq="W")
forecast_holidays = model_holidays.predict(future_holidays)

pred_holidays = forecast_holidays.set_index("ds").loc[test["ds"], "yhat"]
mae_holidays = mean_absolute_error(test["y"], pred_holidays)
mape_holidays = mean_absolute_percentage_error(test["y"], pred_holidays) * 100

# ---- Comparaison finale ----
print("\n=== Comparaison des modèles (MAE / MAPE sur les 12 dernières semaines) ===")
print(f"Modèle baseline      : MAE = {mae_baseline:.2f} | MAPE = {mape_baseline:.2f}%")
print(f"Modèle avec holidays : MAE = {mae_holidays:.2f} | MAPE = {mape_holidays:.2f}%")

if mae_holidays < mae_baseline:
    print("\n-> Le modèle avec holidays est plus précis.")
else:
    print("\n-> Le modèle baseline est aussi bon, voire meilleur.")