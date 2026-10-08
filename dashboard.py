import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from crm_connector import get_connecteur_actif


# ============================================================
# CONFIGURATION
# ============================================================

CHANGEPOINT_PRIOR_SCALE = 0.05
SEASONALITY_PRIOR_SCALE = 10.0

METIERS = [
    "Dev Fullstack",
    "Data/Cloud",
    "DevOps"
]

st.set_page_config(
    page_title="Prévision du Bench",
    layout="wide"
)


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def charger_serie_globale():
    """
    Charge la série globale avec les congés.

    Le fichier doit contenir :
    ds
    bench_total
    bench_disponible
    bench_indisponible
    """

    df = pd.read_csv("bench_weekly_avec_conges.csv")

    df["ds"] = pd.to_datetime(df["ds"])

    colonnes_obligatoires = [
        "ds",
        "bench_total",
        "bench_disponible",
        "bench_indisponible"
    ]

    colonnes_absentes = [
        colonne
        for colonne in colonnes_obligatoires
        if colonne not in df.columns
    ]

    if colonnes_absentes:
        raise ValueError(
            f"Colonnes absentes de bench_weekly_avec_conges.csv : "
            f"{colonnes_absentes}"
        )

    # Prophet utilisera le bench disponible comme cible
    df["y"] = df["bench_disponible"]

    return df


@st.cache_data
def charger_serie_metier(metier):
    """
    Charge la série d'un métier.

    Dans bench_weekly_par_metier.csv :
    - la colonne du métier = bench disponible ;
    - metier_bench_total = bench total ;
    - metier_bench_indisponible = bench en congé.
    """

    df = pd.read_csv("bench_weekly_par_metier.csv")
    df["ds"] = pd.to_datetime(df["ds"])

    if metier not in df.columns:
        raise ValueError(
            f"La colonne du métier '{metier}' est absente du fichier."
        )

    resultat = df[["ds", metier]].copy()

    resultat = resultat.rename(
        columns={
            metier: "y"
        }
    )

    resultat["bench_disponible"] = resultat["y"]

    colonne_total = f"{metier}_bench_total"
    colonne_indisponible = f"{metier}_bench_indisponible"

    if colonne_total in df.columns:
        resultat["bench_total"] = df[colonne_total]
    else:
        resultat["bench_total"] = resultat["y"]

    if colonne_indisponible in df.columns:
        resultat["bench_indisponible"] = df[
            colonne_indisponible
        ]
    else:
        resultat["bench_indisponible"] = 0

    return resultat


@st.cache_data
def charger_toutes_les_series_metier():
    """Prépare les séries disponibles par métier."""

    df = pd.read_csv("bench_weekly_par_metier.csv")
    df["ds"] = pd.to_datetime(df["ds"])

    metiers_disponibles = [
        metier
        for metier in METIERS
        if metier in df.columns
    ]

    resultat = df.melt(
        id_vars="ds",
        value_vars=metiers_disponibles,
        var_name="metier",
        value_name="bench_disponible"
    )

    return resultat


@st.cache_data
def charger_taux_activite(metier):
    """Charge le taux d'activité d'un métier."""

    df = pd.read_csv(
        "bench_weekly_par_metier_avec_activite.csv"
    )

    colonne_activite = f"{metier}_en_activite"
    colonne_taux = f"{metier}_taux_activite"

    if (
        colonne_activite not in df.columns
        or colonne_taux not in df.columns
    ):
        return None

    df["ds"] = pd.to_datetime(df["ds"])

    resultat = df[
        [
            "ds",
            colonne_activite,
            colonne_taux
        ]
    ].copy()

    resultat = resultat.rename(
        columns={
            colonne_activite: "en_activite",
            colonne_taux: "taux_activite"
        }
    )

    return resultat


def charger_registre_brut(fichier_uploade):
    """Charge le registre Excel."""

    if fichier_uploade is not None:
        return pd.read_excel(fichier_uploade)

    return pd.read_excel(
        "Sujet_Bench_enrichi_complet.xlsx"
    )


# ============================================================
# 2. RÉGRESSEUR OPPORTUNITÉS
# ============================================================

def charger_volume_opportunites(
    registre_brut,
    metier=None
):
    """Calcule le volume hebdomadaire d'opportunités."""

    df = registre_brut.copy()

    if "date_entree_bench" not in df.columns:
        return pd.DataFrame(
            columns=["ds", "volume_opportunites"]
        )

    if "volume_opportunites_commerciales" not in df.columns:
        return pd.DataFrame(
            columns=["ds", "volume_opportunites"]
        )

    df["date_entree_bench"] = pd.to_datetime(
        df["date_entree_bench"],
        errors="coerce"
    )

    if metier is not None and "metier" in df.columns:
        df = df[df["metier"] == metier]

    df = df.dropna(
        subset=["date_entree_bench"]
    )

    if df.empty:
        return pd.DataFrame(
            columns=["ds", "volume_opportunites"]
        )

    volume_hebdo = (
        df.set_index("date_entree_bench")
        ["volume_opportunites_commerciales"]
        .resample("W")
        .mean()
        .reset_index()
    )

    volume_hebdo.columns = [
        "ds",
        "volume_opportunites"
    ]

    return volume_hebdo


def simuler_volume(date):
    """Estime le volume futur via le connecteur CRM."""

    try:
        connecteur = get_connecteur_actif()

        volume = connecteur.get_volume_opportunites_hebdo(
            date.date(),
            date.date()
        )

        if len(volume) == 0:
            return 50.0

        return float(
            volume["volume_opportunites"].iloc[0]
        )

    except Exception:
        return 50.0


# ============================================================
# 3. JOURS PARTICULIERS
# ============================================================

def construire_holidays(annee_min, annee_max):
    """Construit les périodes de vacances et fêtes."""

    annees = range(
        int(annee_min),
        int(annee_max) + 1
    )

    vacances_ete = pd.DataFrame({
        "holiday": "vacances_ete",
        "ds": pd.to_datetime([
            f"{annee}-08-01"
            for annee in annees
        ]),
        "lower_window": -15,
        "upper_window": 15
    })

    fetes_fin_annee = pd.DataFrame({
        "holiday": "fetes_fin_annee",
        "ds": pd.to_datetime([
            f"{annee}-12-25"
            for annee in annees
        ]),
        "lower_window": -10,
        "upper_window": 10
    })

    return pd.concat(
        [
            vacances_ete,
            fetes_fin_annee
        ],
        ignore_index=True
    )


# ============================================================
# 4. PRÉPARATION DU DATASET
# ============================================================

def preparer_dataset(vue, registre_brut):
    """Prépare le dataset utilisé par Prophet."""

    if vue == "Vue globale":
        serie = charger_serie_globale()
        metier = None
    else:
        serie = charger_serie_metier(vue)
        metier = vue

    volume = charger_volume_opportunites(
        registre_brut,
        metier
    )

    df = serie.merge(
        volume,
        on="ds",
        how="left"
    )

    if "volume_opportunites" not in df.columns:
        df["volume_opportunites"] = 50.0

    df["volume_opportunites"] = (
        df["volume_opportunites"]
        .interpolate()
        .bfill()
        .ffill()
        .fillna(50.0)
    )

    return df


# ============================================================
# 5. MODÈLE PROPHET
# ============================================================

@st.cache_resource
def entrainer_modele(df, annee_max_future):
    """Entraîne Prophet sur le bench disponible."""

    annee_min = int(df["ds"].dt.year.min())

    holidays = construire_holidays(
        annee_min,
        annee_max_future
    )

    model = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=CHANGEPOINT_PRIOR_SCALE,
        seasonality_prior_scale=SEASONALITY_PRIOR_SCALE
    )

    model.add_seasonality(
        name="mensuelle",
        period=30.5,
        fourier_order=5
    )

    model.add_regressor(
        "volume_opportunites"
    )

    model.fit(
        df[
            [
                "ds",
                "y",
                "volume_opportunites"
            ]
        ]
    )

    return model


def predire(model, df, horizon_semaines):
    """Produit les prévisions."""

    future = model.make_future_dataframe(
        periods=horizon_semaines,
        freq="W"
    )

    future = future.merge(
        df[
            [
                "ds",
                "volume_opportunites"
            ]
        ],
        on="ds",
        how="left"
    )

    future["volume_opportunites"] = future.apply(
        lambda ligne: (
            simuler_volume(ligne["ds"])
            if pd.isna(ligne["volume_opportunites"])
            else ligne["volume_opportunites"]
        ),
        axis=1
    )

    return model.predict(
        future[
            [
                "ds",
                "volume_opportunites"
            ]
        ]
    )


@st.cache_data
def calculer_mae(df):
    """Calcule la MAE par validation croisée."""

    try:
        annee_min = int(df["ds"].dt.year.min())
        annee_max = int(df["ds"].dt.year.max())

        holidays = construire_holidays(
            annee_min,
            annee_max
        )

        model = Prophet(
            holidays=holidays,
            yearly_seasonality=True,
            weekly_seasonality=False,
            changepoint_prior_scale=CHANGEPOINT_PRIOR_SCALE,
            seasonality_prior_scale=SEASONALITY_PRIOR_SCALE
        )

        model.add_seasonality(
            name="mensuelle",
            period=30.5,
            fourier_order=5
        )

        model.add_regressor(
            "volume_opportunites"
        )

        model.fit(
            df[
                [
                    "ds",
                    "y",
                    "volume_opportunites"
                ]
            ]
        )

        df_cv = cross_validation(
            model,
            initial="365 days",
            period="90 days",
            horizon="90 days",
            disable_tqdm=True
        )

        metrics = performance_metrics(df_cv)

        return round(
            metrics["mae"].mean(),
            3
        )

    except Exception:
        return None


# ============================================================
# 6. INTERFACE SIDEBAR
# ============================================================

st.sidebar.header("Filtres")

vue_choisie = st.sidebar.selectbox(
    "Vue",
    ["Vue globale"] + METIERS
)

horizon = st.sidebar.slider(
    "Horizon de prévision (semaines)",
    min_value=4,
    max_value=13,
    value=12
)

st.sidebar.markdown("---")

st.sidebar.header("Données")

fichier_uploade = st.sidebar.file_uploader(
    "Registre brut alternatif",
    type=["xlsx"]
)

st.sidebar.markdown("---")

st.sidebar.header("Alerte")

seuil_alerte = st.sidebar.number_input(
    "Seuil d'alerte",
    min_value=0.0,
    value=8.0,
    step=0.5
)


# ============================================================
# 7. PIPELINE PRINCIPAL
# ============================================================

try:
    registre_brut = charger_registre_brut(
        fichier_uploade
    )

    df = preparer_dataset(
        vue_choisie,
        registre_brut
    )

except Exception as erreur:
    st.error(
        f"Erreur lors du chargement des données : {erreur}"
    )
    st.stop()


annee_max_future = (
    df["ds"].max()
    + pd.Timedelta(weeks=horizon)
).year

model = entrainer_modele(
    df,
    annee_max_future
)

forecast = predire(
    model,
    df,
    horizon
)


# ============================================================
# 8. TITRE ET KPI
# ============================================================

st.title("📊 Prévision du Bench (Intercontrat)")



st.caption(
    f"Vue actuelle : **{vue_choisie}**"
)

valeur_actuelle = float(
    df["bench_disponible"].iloc[-1]
)

derniere_prevision = forecast.iloc[-1]

valeur_prevue = float(
    derniere_prevision["yhat"]
)

bench_total_actuel = float(
    df["bench_total"].iloc[-1]
)

bench_indisponible_actuel = float(
    df["bench_indisponible"].iloc[-1]
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Bench total",
        f"{bench_total_actuel:.1f}"
    )

with col2:
    st.metric(
        "Bench disponible",
        f"{valeur_actuelle:.1f}"
    )

with col3:
    st.metric(
        "Bench indisponible - congés",
        f"{bench_indisponible_actuel:.1f}"
    )

with col4:
    st.metric(
        f"Prévision dans {horizon} semaines",
        f"{valeur_prevue:.1f}",
        delta=f"{valeur_prevue - valeur_actuelle:+.1f}"
    )

st.info(
    "Le bench total regroupe les consultants sans mission. "
    "Le bench disponible exclut les consultants en congé. "
    "Le bench indisponible correspond aux consultants en bench "
    "mais temporairement absents."
)


# ============================================================
# 9. ALERTE
# ============================================================

if valeur_prevue >= seuil_alerte:
    st.error(
        f"🔴 Alerte : le bench disponible prévu "
        f"atteint {valeur_prevue:.1f} consultants."
    )
elif valeur_prevue >= 0.7 * seuil_alerte:
    st.warning(
        f"🟠 À surveiller : le bench disponible prévu "
        f"atteint {valeur_prevue:.1f} consultants."
    )
else:
    st.success(
        f"🟢 Le bench disponible prévu reste sous contrôle "
        f"({valeur_prevue:.1f} consultants)."
    )


# ============================================================
# 10. TAUX D'ACTIVITÉ PAR MÉTIER
# ============================================================

if vue_choisie != "Vue globale":
    taux = charger_taux_activite(vue_choisie)

    if taux is not None and not taux.empty:
        taux_actuel = float(
            taux["taux_activite"].iloc[-1]
        )

        st.metric(
            f"Taux d'activité - {vue_choisie}",
            f"{taux_actuel:.1f}%"
        )


# ============================================================
# 11. GRAPHIQUE PRINCIPAL
# ============================================================

st.subheader(
    "Évolution du bench et impact des congés"
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["ds"],
        y=df["bench_total"],
        mode="lines+markers",
        name="Bench total",
        line=dict(color="#1f77b4")
    )
)

fig.add_trace(
    go.Scatter(
        x=df["ds"],
        y=df["bench_disponible"],
        mode="lines+markers",
        name="Bench disponible",
        line=dict(color="#2ca02c")
    )
)

fig.add_trace(
    go.Scatter(
        x=df["ds"],
        y=df["bench_indisponible"],
        mode="lines+markers",
        name="Bench indisponible - congés",
        line=dict(color="#d62728")
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Prévision du bench disponible",
        line=dict(
            color="#ff7f0e",
            dash="dash"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=pd.concat(
            [
                forecast["ds"],
                forecast["ds"][::-1]
            ]
        ),
        y=pd.concat(
            [
                forecast["yhat_upper"],
                forecast["yhat_lower"][::-1]
            ]
        ),
        fill="toself",
        fillcolor="rgba(255,127,14,0.15)",
        line=dict(
            color="rgba(255,255,255,0)"
        ),
        name="Intervalle de confiance"
    )
)

fig.add_hline(
    y=seuil_alerte,
    line_dash="dot",
    line_color="red",
    annotation_text="Seuil d'alerte",
    annotation_position="top left"
)

fig.update_layout(
    title=f"Bench total, disponible et congés - {vue_choisie}",
    xaxis_title="Semaine",
    yaxis_title="Nombre moyen de consultants",
    hovermode="x unified",
    height=550
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 12. COMPARAISON DES MÉTIERS
# ============================================================

st.markdown("---")
st.subheader("Bench disponible par métier")

try:
    df_metiers = charger_toutes_les_series_metier()

    fig_metiers = go.Figure()

    couleurs = {
        "Dev Fullstack": "#1f77b4",
        "Data/Cloud": "#2ca02c",
        "DevOps": "#d62728"
    }

    for metier in df_metiers["metier"].unique():
        sous_df = df_metiers[
            df_metiers["metier"] == metier
        ]

        fig_metiers.add_trace(
            go.Scatter(
                x=sous_df["ds"],
                y=sous_df["bench_disponible"],
                mode="lines",
                name=metier,
                line=dict(
                    color=couleurs.get(
                        metier,
                        "#555555"
                    )
                )
            )
        )

    fig_metiers.update_layout(
        title="Évolution du bench disponible par métier",
        xaxis_title="Semaine",
        yaxis_title="Bench disponible",
        hovermode="x unified",
        height=400
    )

    st.plotly_chart(
        fig_metiers,
        use_container_width=True
    )

except Exception as erreur:
    st.warning(
        f"Comparaison par métier indisponible : {erreur}"
    )


# ============================================================
# 13. TABLEAU DES DONNÉES
# ============================================================

with st.expander("Voir les données historiques avec les congés"):

    tableau = df[
        [
            "ds",
            "bench_total",
            "bench_disponible",
            "bench_indisponible"
        ]
    ].copy()

    tableau["ds"] = tableau["ds"].dt.strftime(
        "%Y-%m-%d"
    )

    st.dataframe(
        tableau,
        use_container_width=True
    )


with st.expander("Voir les prévisions"):

    table_previsions = forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ].tail(horizon).copy()

    table_previsions = table_previsions.rename(
        columns={
            "ds": "Semaine",
            "yhat": "Prévision bench disponible",
            "yhat_lower": "Borne basse",
            "yhat_upper": "Borne haute"
        }
    )

    st.dataframe(
        table_previsions,
        use_container_width=True
    )


# ============================================================
# 14. ÉVALUATION
# ============================================================

with st.expander("Évaluation du modèle"):

    mae = calculer_mae(df)

    if mae is not None:
        st.metric(
            "MAE du modèle",
            f"{mae:.3f}"
        )

        st.caption(
            "La MAE est calculée sur la cible "
            "« bench disponible »."
        )
    else:
        st.warning(
            "L'évaluation n'est pas disponible. "
            "La série est peut-être trop courte."
        )


# ============================================================
# 15. EXPORT CSV
# ============================================================

st.markdown("---")
st.subheader("Exports")

csv_historique = df[
    [
        "ds",
        "bench_total",
        "bench_disponible",
        "bench_indisponible"
    ]
].to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Télécharger les données avec congés",
    data=csv_historique,
    file_name="bench_avec_conges_dashboard.csv",
    mime="text/csv"
)

csv_prevision = forecast[
    [
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper"
    ]
].to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Télécharger les prévisions",
    data=csv_prevision,
    file_name="previsions_bench_disponible.csv",
    mime="text/csv"
)
