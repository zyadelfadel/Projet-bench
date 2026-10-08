
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Charger la série déjà transformée
weekly = pd.read_csv("bench_weekly.csv")
weekly["ds"] = pd.to_datetime(weekly["ds"])

# Créer et entraîner le modèle Prophet (baseline, sans réglages)
model = Prophet()
model.fit(weekly)

# Créer un calendrier futur de 12 semaines (~3 mois) après la dernière date connue
future = model.make_future_dataframe(periods=12, freq="W")

# Générer les prédictions
forecast = model.predict(future)

# Afficher les colonnes principales du résultat
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(15))

# Visualiser la prévision
model.plot(forecast)
plt.title("Prévision du bench - Modèle Prophet baseline")
plt.show()