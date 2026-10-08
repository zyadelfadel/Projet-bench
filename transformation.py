import pandas as pd
df= pd.read_excel("Sujet_Bench_enrichi_complet.xlsx")

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


def preparer_donnees(fichier_entree):
    # Lecture du fichier Excel
    df = pd.read_excel(fichier_entree)

    # Conversion des dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_debut_conge"] = pd.to_datetime(
        df["date_debut_conge"],
        errors="coerce"
    )
    df["date_fin_conge"] = pd.to_datetime(
        df["date_fin_conge"],
        errors="coerce"
    )

    # Consultant en congé à la date considérée
    df["en_conge"] = (
        (df["date"] >= df["date_debut_conge"]) &
        (df["date"] <= df["date_fin_conge"])
    ).astype(int)

    # Exemple : adaptation selon ta colonne de statut
    df["en_mission"] = (
        df["statut"].str.lower() == "en mission"
    ).astype(int)

    # Bench total : consultant qui n'est pas en mission
    df["bench_total"] = (
        df["en_mission"] == 0
    ).astype(int)

    # Bench disponible : sans mission et pas en congé
    df["bench_disponible"] = (
        (df["en_mission"] == 0) &
        (df["en_conge"] == 0)
    ).astype(int)

    # Bench indisponible : sans mission mais en congé
    df["bench_indisponible"] = (
        (df["en_mission"] == 0) &
        (df["en_conge"] == 1)
    ).astype(int)

    return df


