import pandas as pd


df = pd.read_excel("Sujet_Bench_enrichi_complet.xlsx")

print(df.head())

print(df.info())

import pandas as pd

df = pd.read_excel("Sujet_Bench_enrichi_complet.xlsx")


print("Répartition par métier :")
print(df["metier"].value_counts())


print("\nMotifs de sortie :")
print(df["motif_sortie"].value_counts())


print("\nStatistiques durée (en jours) :")
print(df["duree_jours"].describe())


date_range = pd.date_range(df["date_entree_bench"].min(), df["date_sortie_bench"].max(), freq="D")


daily_counts = pd.Series(0, index=date_range)
for _, row in df.iterrows():
    mask = (date_range >= row["date_entree_bench"]) & (date_range <= row["date_sortie_bench"])
    daily_counts[mask] += 1


weekly = daily_counts.resample("W").mean().reset_index()
weekly.columns = ["ds", "y"]

print(weekly.head(15))
print(f"\nNombre de semaines : {len(weekly)}")


weekly.to_csv("bench_weekly.csv", index=False)

import pandas as pd

df = pd.read_excel("Sujet_Bench_enrichi_complet.xlsx")

print(df.columns.tolist())
print(df.head())
print(df.info())
