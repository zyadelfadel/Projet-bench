import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

weekly = pd.read_csv("bench_weekly.csv")
weekly["ds"] = pd.to_datetime(weekly["ds"])

# Holidays (comme avant)
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

# Modèle avec saisonnalité annuelle explicite
model = Prophet(
    holidays=holidays,
    yearly_seasonality=True,
    weekly_seasonality=False,  # pas pertinent, données déjà hebdo
)

# Ajout explicite de la saisonnalité mensuelle
model.add_seasonality(
    name="mensuelle",
    period=30.5,
    fourier_order=5,
)

model.fit(weekly)

future = model.make_future_dataframe(periods=12, freq="W")
forecast = model.predict(future)

model.plot(forecast)
plt.title("Prévision - holidays + saisonnalité annuelle + mensuelle")
plt.show()

model.plot_components(forecast)
plt.show()