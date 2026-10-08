import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Charger la série transformée
weekly = pd.read_csv("bench_weekly.csv")
weekly["ds"] = pd.to_datetime(weekly["ds"])

# Définir les vacances d'été (une ligne par année couverte par tes données)
ete = pd.DataFrame({
    "holiday": "vacances_ete",
    "ds": pd.to_datetime(["2023-08-01", "2024-08-01", "2025-08-01"]),
    "lower_window": -15,   # l'effet commence 15 jours avant cette date
    "upper_window": 15,    # et continue 15 jours après
})

# Définir les fêtes de fin d'année
fin_annee = pd.DataFrame({
    "holiday": "fetes_fin_annee",
    "ds": pd.to_datetime(["2023-12-25", "2024-12-25", "2025-12-25"]),
    "lower_window": -10,
    "upper_window": 10,
})

# Combiner les deux dans un seul tableau
holidays = pd.concat([ete, fin_annee])

# Créer le modèle en lui passant les holidays
model = Prophet(holidays=holidays)
model.fit(weekly)

# Prédire les 12 prochaines semaines
future = model.make_future_dataframe(periods=12, freq="W")
forecast = model.predict(future)
print(forecast[forecast["vacances_ete"] != 0][["ds", "vacances_ete"]])

# Visualiser
model.plot(forecast)
plt.title("Prévision du bench - Modèle avec holidays")
plt.show()

model.plot_components(forecast)
plt.show()