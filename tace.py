import pandas as pd
import matplotlib.pyplot as plt

# Forcer pandas à tout afficher, sans troncature
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Charger le fichier avec les taux d'activité déjà calculés
weekly = pd.read_csv("bench_weekly_par_metier_avec_activite.csv")
weekly["ds"] = pd.to_datetime(weekly["ds"])

# Liste des colonnes de taux d'activité (celles qui finissent par "_taux_activite")
colonnes_taux = [col for col in weekly.columns if col.endswith("_taux_activite")]

print("=== Taux d'activité (TACE approché) - détail par semaine ===")
print(weekly[["ds"] + colonnes_taux])

print("\n=== Taux d'activité MOYEN sur toute la période, par métier ===")
for col in colonnes_taux:
    metier = col.replace("_taux_activite", "")
    moyenne = weekly[col].mean().round(1)
    print(f"{metier} : {moyenne}%")

# Visualisation : tracé de l'évolution du taux d'activité pour chaque métier
plt.figure(figsize=(12, 6))
for col in colonnes_taux:
    plt.plot(weekly["ds"], weekly[col], label=col.replace("_taux_activite", ""))

plt.xlabel("Semaine")
plt.ylabel("Taux d'activité (%)")
plt.title("Évolution hebdomadaire du Taux d'Activité par Métier")
plt.legend()
plt.grid(True)
plt.show()
