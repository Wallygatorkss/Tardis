import streamlit as st
import pandas as pd
import joblib
from graph import (
    plot_distribution_retards,
    plot_violin_services,
    plot_evolution_retards,
)

df = pd.read_csv("cleaned_dataset.csv", sep=";")
choix_depart = df["Gare de départ"].unique()
choix_depart = choix_depart[(choix_depart != "0")]

garesel = st.sidebar.selectbox("Gare de départ", sorted(choix_depart))
destinations_possibles = df[df["Gare de départ"] == garesel]["Gare d'arrivée"].unique()


st.sidebar.header("Filtres Temporels")

liste_annees = [int(y) for y in df["Annee"].unique() if pd.notna(y) and y != 0]
liste_annees = sorted(set(liste_annees) | {2026, 2027})
annee_sel = st.sidebar.selectbox(
    "Choisir l'année", liste_annees, index=len(liste_annees) - 1
)

mois_sel = st.sidebar.slider("Choisir le mois", 1, 12, (1, 12))


data_filter = df[
    (df["Gare de départ"] == garesel)
    & (df["Annee"] == annee_sel)
    & (df["Mois"] >= mois_sel[0])
    & (df["Mois"] <= mois_sel[1])
].copy()

st.header("Statistiques")

col1, col2, col3 = st.columns(3)

if data_filter.empty:
    with col1:
        st.metric(label="Retard Moyen", value="N/A")
    with col2:
        st.metric(label="Somme de la circulation", value=0)
    with col3:
        st.metric(label="Pourcentage de ponctualité", value="N/A")
else:
    moyenne_retard = data_filter["Retard moyen des trains en retard à l'arrivée"].mean()

    total_prevu = data_filter["Nombre de circulations prévues"].sum()
    total_annules = data_filter["Nombre de trains annulés"].sum()
    train_reel = total_prevu - total_annules
    if train_reel > 0:
        total_retards = data_filter["Nombre de trains en retard à l'arrivée"].sum()
        ponctualite = ((train_reel - total_retards) / train_reel) * 100
    else:
        ponctualite = 0

    with col1:
        st.metric(label="Retard Moyen", value=f"{moyenne_retard:.1f} min")
    with col2:
        st.metric(label="Somme de la circulation", value=int(total_prevu))
    with col3:
        st.metric(label="Pourcentage de ponctualité", value=f"{ponctualite:.1f} %")

st.divider()
st.header(f"Distribution des retards : {garesel}")

if not data_filter.empty:
    chart_df = (
        data_filter.groupby("Date")["Retard moyen des trains en retard à l'arrivée"]
        .mean()
        .reset_index()
    )
    chart_df["Date"] = chart_df["Date"].astype(str)

    if not chart_df.empty:
        st.line_chart(
            data=chart_df, x="Date", y="Retard moyen des trains en retard à l'arrivée"
        )

        st.subheader("Répartition des durées de retard")
        distrib = (
            data_filter["Retard moyen des trains en retard à l'arrivée"]
            .round()
            .value_counts()
            .reset_index()
        )
        distrib.columns = ["Durée Retard", "Nombre de Trains"]
        st.line_chart(data=distrib, x="Durée Retard", y="Nombre de Trains")
    else:
        st.warning("Données insuffisantes pour générer les graphiques.")
else:
    st.info(
        f"Aucune donnée disponible pour {garesel} en {annee_sel} (mois {mois_sel[0]}–{mois_sel[1]})."
    )
st.divider()

target_col = "Retard moyen de tous les trains à l'arrivée"

plot_distribution_retards(data_filter, target_col)

plot_violin_services(data_filter, target_col)

plot_evolution_retards(data_filter, target_col)

st.divider()
st.header("Simulateur de Prédiction")

c1, c2 = st.columns(2)

with c1:
    gare_arr = st.selectbox("Destination", sorted(destinations_possibles))

with c2:
    modele_choisi = st.sidebar.selectbox(
        "Choisir le modèle d'IA",
        [
            "Régression Linéaire 50%",
            "Arbre de Décision Huge 76%",
            "Arbre de Décision Low 29%",
            "XGBoost 75%",
        ],
    )

fichiers_modeles = {
    "Régression Linéaire 50%": "model.pkl",
    "Arbre de Décision Huge 76%": "modele_random_tree_huge.pkl",
    "Arbre de Décision Low 29%": "modele_random_tree_low.pkl",
    "XGBoost 75%": "modele_XGB_Regressor.pkl",
}

nom = fichiers_modeles[modele_choisi]
model_ia = joblib.load(nom)

duree_moy = df[(df["Gare de départ"] == garesel) & (df["Gare d'arrivée"] == gare_arr)][
    "Durée moyenne du trajet"
].mean()
duree_moy = df[(df["Gare de départ"] == garesel) & (df["Gare d'arrivée"] == gare_arr)][
    "Durée moyenne du trajet"
].mean()

if st.button("Lancer la prédiction"):
    try:
        stats_trajet = df[
            (df["Gare de départ"] == garesel) & (df["Gare d'arrivée"] == gare_arr)
        ]

        if stats_trajet.empty:
            st.error("Données historiques manquantes.")
        else:
            moyens = stats_trajet.mean(numeric_only=True)

            service_val = moyens.get("Service", 0)
            saison_val = moyens.get("Saison", 0)

            input_dict = {
                "Annee": [annee_sel],
                "Durée moyenne du trajet": [moyens.get("Durée moyenne du trajet", 0)],
                "Service": [service_val],
                "Mois": [mois_sel[0]],
                "Saison": [saison_val],
                "Retard_historique_ligne": [
                    moyens.get("Retard moyen de tous les trains à l'arrivée", 0)
                ],
                "Retard_historique_ligne_mois": [
                    moyens.get("Retard moyen des trains en retard à l'arrivée", 0)
                ],
                "Retard_moyen_dep_histo": [
                    moyens.get("Retard moyen de tous les trains au départ", 0)
                ],
                "Annules_histo": [moyens.get("Nombre de trains annulés", 0)],
                "Nombre de circulations prévues": [
                    moyens.get("Nombre de circulations prévues", 0)
                ],
                "Nombre de trains annulés": [moyens.get("Nombre de trains annulés", 0)],
                "Nombre de trains en retard au départ": [
                    moyens.get("Nombre de trains en retard au départ", 0)
                ],
                "Retard moyen des trains en retard au départ": [
                    moyens.get("Retard moyen des trains en retard au départ", 0)
                ],
                "Retard moyen de tous les trains au départ": [
                    moyens.get("Retard moyen de tous les trains au départ", 0)
                ],
                "Nombre trains en retard > 15min": [
                    moyens.get("Nombre trains en retard > 15min", 0)
                ],
                "Retard moyen trains en retard > 15 (si liaison concurrencée par vol)": [
                    moyens.get(
                        "Retard moyen trains en retard > 15 (si liaison concurrencée par vol)",
                        0,
                    )
                ],
                "Nombre trains en retard > 30min": [
                    moyens.get("Nombre trains en retard > 30min", 0)
                ],
                "Nombre trains en retard > 60min": [
                    moyens.get("Nombre trains en retard > 60min", 0)
                ],
                "Gare d'arrivée encodée": [moyens.get("Gare d'arrivée encodée", 0)],
                "Gare de départ encodée": [moyens.get("Gare de départ encodée", 0)],
            }

            input_df = pd.DataFrame(input_dict)[list(model_ia.feature_names_in_)]

            prediction = model_ia.predict(input_df)
            st.success(f"Résultat : {float(prediction.flatten()[0]):.2f} min")

    except Exception as e:
        st.error(f"Erreur de compatibilité : {e}")
        st.write("Colonnes attendues par le modèle :", list(model_ia.feature_names_in_))
