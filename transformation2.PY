import pandas as pd


FICHIER_EXCEL = "Sujet_Bench_enrichi_complet.xlsx"


def charger_donnees():
    """Charge et prépare les dates de la base Excel."""

    df = pd.read_excel(FICHIER_EXCEL)

    colonnes_dates = [
        "date_entree_bench",
        "date_sortie_bench",
        "date_debut_conge",
        "date_fin_conge"
    ]

    for colonne in colonnes_dates:
        df[colonne] = pd.to_datetime(
            df[colonne],
            errors="coerce"
        )

    # Les dates d'entrée et de sortie du bench sont obligatoires
    df = df.dropna(
        subset=[
            "date_entree_bench",
            "date_sortie_bench"
        ]
    )

    return df


def transformation_globale():
    """
    Calcule le bench hebdomadaire global avec prise en compte des congés.

    bench_total :
        Nombre moyen de consultants présents dans le bench.

    bench_indisponible :
        Consultants présents dans le bench et en congé.

    bench_disponible :
        Consultants présents dans le bench mais pas en congé.
    """

    df = charger_donnees()

    # Période globale d'analyse
    date_range = pd.date_range(
        start=df["date_entree_bench"].min(),
        end=df["date_sortie_bench"].max(),
        freq="D"
    )

    # Séries quotidiennes
    daily_bench_total = pd.Series(
        0.0,
        index=date_range
    )

    daily_bench_indisponible = pd.Series(
        0.0,
        index=date_range
    )

    # Calcul quotidien
    for _, row in df.iterrows():

        # Période pendant laquelle le consultant est dans le bench
        mask_bench = (
            (date_range >= row["date_entree_bench"]) &
            (date_range <= row["date_sortie_bench"])
        )

        daily_bench_total.loc[mask_bench] += 1

        # Vérification des dates de congé
        # Si les dates sont NaT, le consultant n'est pas compté en congé
        if (
            pd.notna(row["date_debut_conge"]) and
            pd.notna(row["date_fin_conge"])
        ):

            # Période de congé
            mask_conge = (
                (date_range >= row["date_debut_conge"]) &
                (date_range <= row["date_fin_conge"])
            )

            # Consultant à la fois dans le bench et en congé
            mask_bench_indisponible = (
                mask_bench & mask_conge
            )

            daily_bench_indisponible.loc[
                mask_bench_indisponible
            ] += 1

    # Bench réellement disponible
    daily_bench_disponible = (
        daily_bench_total - daily_bench_indisponible
    )

    # Création de la série quotidienne
    daily = pd.DataFrame({
        "bench_total": daily_bench_total,
        "bench_disponible": daily_bench_disponible,
        "bench_indisponible": daily_bench_indisponible
    })

    # Agrégation hebdomadaire
    weekly = (
        daily
        .resample("W")
        .mean()
        .reset_index()
    )

    weekly = weekly.rename(
        columns={
            "index": "ds"
        }
    )

    # Arrondi
    colonnes_bench = [
        "bench_total",
        "bench_disponible",
        "bench_indisponible"
    ]

    weekly[colonnes_bench] = weekly[colonnes_bench].round(2)

    # Fichier détaillé
    weekly.to_csv(
        "bench_weekly_avec_conges.csv",
        index=False
    )

    # Fichier utilisé par Prophet
    # La cible y correspond maintenant au bench disponible
    weekly_prophet = weekly[
        [
            "ds",
            "bench_disponible"
        ]
    ].copy()

    weekly_prophet = weekly_prophet.rename(
        columns={
            "bench_disponible": "y"
        }
    )

    weekly_prophet.to_csv(
        "bench_weekly.csv",
        index=False
    )

    print("Transformation globale terminée.")
    print("Fichiers créés :")
    print("- bench_weekly_avec_conges.csv")
    print("- bench_weekly.csv")
    print()
    print(weekly.head(15))
    print()
    print("Colonnes disponibles :")
    print(weekly.columns.tolist())

    return weekly


def transformation_par_metier():
    """
    Calcule le bench disponible, total et indisponible par métier.
    """

    df = charger_donnees()

    date_range = pd.date_range(
        start=df["date_entree_bench"].min(),
        end=df["date_sortie_bench"].max(),
        freq="D"
    )

    metiers = df["metier"].dropna().unique()

    daily_total_par_metier = {}
    daily_indisponible_par_metier = {}
    daily_disponible_par_metier = {}

    for metier in metiers:

        df_metier = df[df["metier"] == metier]

        counts_total = pd.Series(
            0.0,
            index=date_range
        )

        counts_indisponible = pd.Series(
            0.0,
            index=date_range
        )

        for _, row in df_metier.iterrows():

            # Présence dans le bench
            mask_bench = (
                (date_range >= row["date_entree_bench"]) &
                (date_range <= row["date_sortie_bench"])
            )

            counts_total.loc[mask_bench] += 1

            # Présence en congé
            if (
                pd.notna(row["date_debut_conge"]) and
                pd.notna(row["date_fin_conge"])
            ):

                mask_conge = (
                    (date_range >= row["date_debut_conge"]) &
                    (date_range <= row["date_fin_conge"])
                )

                mask_indisponible = (
                    mask_bench & mask_conge
                )

                counts_indisponible.loc[
                    mask_indisponible
                ] += 1

        counts_disponible = (
            counts_total - counts_indisponible
        )

        # Le nom du métier contient la série du bench disponible
        daily_disponible_par_metier[metier] = counts_disponible

        # Séries complémentaires
        daily_total_par_metier[
            f"{metier}_bench_total"
        ] = counts_total

        daily_indisponible_par_metier[
            f"{metier}_bench_indisponible"
        ] = counts_indisponible

    # Le fichier principal conserve les colonnes métier
    # utilisées auparavant par tes modèles
    daily_df = pd.DataFrame(daily_disponible_par_metier)

    weekly_par_metier = (
        daily_df
        .resample("W")
        .mean()
        .reset_index()
        .rename(columns={"index": "ds"})
    )

    # Ajout des colonnes détaillées
    daily_total_df = pd.DataFrame(daily_total_par_metier)

    weekly_total = (
        daily_total_df
        .resample("W")
        .mean()
        .reset_index(drop=True)
    )

    daily_indisponible_df = pd.DataFrame(
        daily_indisponible_par_metier
    )

    weekly_indisponible = (
        daily_indisponible_df
        .resample("W")
        .mean()
        .reset_index(drop=True)
    )

    for colonne in weekly_total.columns:
        weekly_par_metier[colonne] = weekly_total[colonne].round(2)

    for colonne in weekly_indisponible.columns:
        weekly_par_metier[colonne] = (
            weekly_indisponible[colonne].round(2)
        )

    weekly_par_metier.to_csv(
        "bench_weekly_par_metier.csv",
        index=False
    )

    print("Transformation par métier terminée.")
    print("- bench_weekly_par_metier.csv")
    print()
    print(weekly_par_metier.head(15))
    print()
    print("Colonnes disponibles :")
    print(weekly_par_metier.columns.tolist())

    return weekly_par_metier


def transformation_avec_activite():
    """
    Calcule le nombre de consultants en activité
    et le taux d'activité approximatif par métier.
    """

    effectifs = pd.read_csv("effectifs_metiers.csv")

    weekly_par_metier = pd.read_csv(
        "bench_weekly_par_metier.csv"
    )

    weekly_par_metier["ds"] = pd.to_datetime(
        weekly_par_metier["ds"]
    )

    for _, row in effectifs.iterrows():

        metier = row["metier"]
        effectif = row["effectif_total"]

        # Les colonnes métier représentent maintenant
        # le bench disponible
        if metier in weekly_par_metier.columns:

            weekly_par_metier[
                f"{metier}_en_activite"
            ] = (
                effectif - weekly_par_metier[metier]
            )

            weekly_par_metier[
                f"{metier}_taux_activite"
            ] = (
                weekly_par_metier[
                    f"{metier}_en_activite"
                ]
                / effectif
                * 100
            ).round(1)

    weekly_par_metier.to_csv(
        "bench_weekly_par_metier_avec_activite.csv",
        index=False
    )

    print("Transformation avec activité terminée.")
    print("- bench_weekly_par_metier_avec_activite.csv")
    print()
    print(weekly_par_metier.head(15))
    print()
    print("Colonnes disponibles :")
    print(weekly_par_metier.columns.tolist())

    return weekly_par_metier


# Zone d'exécution
if __name__ == "__main__":

    transformation_globale()
    transformation_par_metier()
    transformation_avec_activite()
