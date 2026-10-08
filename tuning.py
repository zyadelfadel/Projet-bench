import pandas as pd
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Charger la série cible
weekly = pd.read_csv("bench_weekly.csv")
weekly["ds"] = pd.to_datetime(weekly["ds"])

# Grille de valeurs à tester
changepoint_scales = [0.01, 0.05, 0.1, 0.5]
seasonality_scales = [0.1, 1.0, 10.0]

# Toutes les combinaisons possibles (produit croisé)
combinaisons = list(itertools.product(changepoint_scales, seasonality_scales))
print(f"Nombre de combinaisons à tester : {len(combinaisons)}")

resultats = []

for cp_scale, season_scale in combinaisons:
    model = Prophet(
        changepoint_prior_scale=cp_scale,
        seasonality_prior_scale=season_scale,
    )
    model.fit(weekly)

    df_cv = cross_validation(
        model,
        initial="365 days",
        period="90 days",
        horizon="90 days",
        disable_tqdm=True,
    )
    df_metrics = performance_metrics(df_cv)
    mae_moyen = df_metrics["mae"].mean()
    mape_moyen = df_metrics["mape"].mean() * 100

    resultats.append({
        "changepoint_prior_scale": cp_scale,
        "seasonality_prior_scale": season_scale,
        "mae_moyen": round(mae_moyen, 3),
        "mape_moyen": round(mape_moyen, 2),
    })
    print(f"changepoint={cp_scale}, seasonality={season_scale} -> MAE = {mae_moyen:.3f} | MAPE = {mape_moyen:.2f}%")

# Trier les résultats du meilleur au moins bon
resultats_df = pd.DataFrame(resultats).sort_values("mae_moyen")
print("\n=== Résultats triés (meilleur en premier) ===")
print(resultats_df)

meilleure_combo = resultats_df.iloc[0]
print(f"\nMeilleure combinaison : changepoint_prior_scale={meilleure_combo['changepoint_prior_scale']}, "
      f"seasonality_prior_scale={meilleure_combo['seasonality_prior_scale']} "
      f"(MAE = {meilleure_combo['mae_moyen']}, MAPE = {meilleure_combo['mape_moyen']}%)")