import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Charger la série cible (ds, y)
weekly = pd.read_csv("bench_weekly.csv")
weekly["ds"] = pd.to_datetime(weekly["ds"])

# 2. Construire la série hebdomadaire du régresseur (volume d'opportunités commerciales)
df = pd.read_excel("Sujet_Bench_enrichi_complet.xlsx")
df["date_entree_bench"] = pd.to_datetime(df["date_entree_bench"])

# Moyenne du volume d'opportunités par semaine d'entrée en bench
volume_hebdo = (
    df.set_index("date_entree_bench")["volume_opportunites_commerciales"]
    .resample("W")
    .mean()
    .reset_index()
)
volume_hebdo.columns = ["ds", "volume_opportunites"]

# 3. Fusionner avec la série cible
weekly = weekly.merge(volume_hebdo, on="ds", how="left")

# Certaines semaines n'ont pas eu de nouvelle entrée bench -> valeur manquante
# On interpole pour combler les trous (moyenne entre les points connus)
weekly["volume_opportunites"] = weekly["volume_opportunites"].interpolate()
weekly["volume_opportunites"] = weekly["volume_opportunites"].bfill().ffill()

print(weekly.head(15))

# 4. Créer et entraîner le modèle avec le régresseur
model = Prophet()
model.add_regressor("volume_opportunites")
model.fit(weekly)

# 5. Préparer le futur AVEC des valeurs de régresseur
future = model.make_future_dataframe(periods=12, freq="W")

# Pour les 12 semaines futures, on simule le volume selon le mois (même logique que la génération)
def simuler_volume(date):
    mois = date.month
    base = 50
    if mois in (8, 12):
        base -= 20
    elif mois in (1, 9):
        base += 15
    return base

# On remplit le régresseur pour tout le futur dataframe
future = future.merge(weekly[["ds", "volume_opportunites"]], on="ds", how="left")
future["volume_opportunites"] = future.apply(
    lambda row: simuler_volume(row["ds"]) if pd.isna(row["volume_opportunites"]) else row["volume_opportunites"],
    axis=1
)

# 6. Prédire
forecast = model.predict(future)

model.plot(forecast)
plt.title("Prévision du bench - avec régresseur (volume opportunités)")
plt.show()

model.plot_components(forecast)
plt.show()