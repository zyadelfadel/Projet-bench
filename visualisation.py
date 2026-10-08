import pandas as pd
import matplotlib.pyplot as plt


weekly = pd.read_csv("bench_weekly.csv")
weekly["ds"] = pd.to_datetime(weekly["ds"])


plt.figure(figsize=(12, 5))
plt.plot(weekly["ds"], weekly["y"], marker="o", markersize=3)
plt.title("Évolution hebdomadaire du nombre de consultants au bench")
plt.xlabel("Date")
plt.ylabel("Nombre moyen au bench")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()