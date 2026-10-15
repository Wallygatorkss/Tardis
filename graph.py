import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np


def plot_distribution_retards(df_clean, target_col):

    series = df_clean[target_col].dropna()

    if series.empty:
        st.info("Aucune donnée disponible pour ce graphique.")
        return

    q_low, q_high = series.quantile([0.01, 0.99]).tolist()
    mean_val = float(series.mean())
    median_val = float(series.median())

    fig, ax = plt.subplots()  # pas de figsize fixe

    sns.histplot(series, bins=30, kde=True, ax=ax)

    ax.axvline(mean_val, linestyle="--", linewidth=2, label=f"Moyenne: {mean_val:.2f}")
    ax.axvline(
        median_val, linestyle="-", linewidth=2, label=f"Médiane: {median_val:.2f}"
    )

    ax.set_title("Distribution des retards à l'arrivée")
    ax.set_xlabel("Retard (minutes)")
    ax.set_ylabel("Nombre d'observations")
    ax.set_xlim(q_low, q_high)
    ax.legend()

    st.pyplot(fig, use_container_width=True)


def plot_violin_services(df_clean, target_col):

    if target_col not in df_clean.columns:
        st.warning(f"Colonne cible introuvable: {target_col}")
        return

    if "Service" not in df_clean.columns:
        st.warning(
            "Colonne 'Service' absente : impossible de tracer la comparaison par service."
        )
        return

    plot_df = df_clean[["Service", target_col]].dropna().copy()

    if plot_df.empty:
        st.info("Aucune donnée disponible pour ce graphique.")
        return

    service_stats = (
        plot_df.groupby("Service")[target_col]
        .agg(n="count", mean="mean", median="median")
        .sort_values("mean", ascending=False)
    )

    services_order = service_stats.index.tolist()

    x_low, x_high = plot_df[target_col].quantile([0.01, 0.99]).tolist()
    x_pad = (x_high - x_low) * 0.08

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(services_order))))

    sns.violinplot(
        data=plot_df,
        y="Service",
        x=target_col,
        order=services_order,
        cut=0,
        inner="quartile",
        density_norm="width",
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        y="Service",
        x=target_col,
        order=services_order,
        color="black",
        alpha=0.25,
        size=2,
        jitter=0.25,
        ax=ax,
    )

    y_pos = np.arange(len(services_order))
    ax.scatter(
        service_stats["mean"].to_numpy(),
        y_pos,
        marker="D",
        s=40,
        color="tab:orange",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="Moyenne",
    )
    ax.scatter(
        service_stats["median"].to_numpy(),
        y_pos,
        marker="o",
        s=28,
        color="tab:green",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="Médiane",
    )

    x_text = x_high + x_pad * 0.15
    for i, svc in enumerate(services_order):
        ax.text(
            x_text, i, f"n={int(service_stats.loc[svc, 'n'])}", va="center", fontsize=9
        )

    ax.set_xlim(x_low, x_high + x_pad)
    ax.set_title(f"Retards par service — distribution + points\nVariable: {target_col}")
    ax.set_xlabel("Retard (minutes)")
    ax.set_ylabel("Service")
    ax.legend(loc="lower right")

    plt.tight_layout(pad=0.6)
    st.pyplot(fig, use_container_width=True)


def plot_evolution_retards(df_clean, target_col):

    if "Date" not in df_clean.columns:
        st.warning(
            "Colonne 'Date' absente : impossible d'afficher l'évolution temporelle."
        )
        return

    if df_clean.empty:
        st.info("Aucune donnée disponible pour cette période.")
        return

    if "Service" in df_clean.columns:
        trend_df = (
            df_clean.groupby(["Date", "Service"], as_index=False)[target_col]
            .mean()
            .sort_values("Date")
        )

        if trend_df.empty:
            st.info("Aucune donnée exploitable pour cette évolution.")
            return

        fig, ax = plt.subplots()

        sns.lineplot(data=trend_df, x="Date", y=target_col, hue="Service", ax=ax)

        ax.set_title("Évolution mensuelle des retards à l'arrivée (par service)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Minutes")

    else:
        trend_df = (
            df_clean.groupby("Date", as_index=False)[target_col]
            .mean()
            .sort_values("Date")
        )

        if trend_df.empty:
            st.info("Aucune donnée exploitable pour cette évolution.")
            return

        fig, ax = plt.subplots()

        sns.lineplot(data=trend_df, x="Date", y=target_col, ax=ax)

        ax.set_title("Évolution mensuelle des retards à l'arrivée")
        ax.set_xlabel("Date")
        ax.set_ylabel("Minutes")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)


def plot_top_routes_adapted(df_clean, target_col, min_n=50, top_k=15):
    routes = df_clean[["Gare de départ", "Gare d'arrivée", target_col]].dropna().copy()
    routes["Route"] = routes["Gare de départ"] + " → " + routes["Gare d'arrivée"]

    route_stats = (
        routes.groupby("Route")[target_col]
        .agg(count="count", mean="mean")
        .sort_values("mean", ascending=False)
    )

    # Adapter min_n si dataset petit
    actual_min_n = min(min_n, route_stats["count"].max())
    route_stats_f = route_stats[route_stats["count"] >= actual_min_n]

    if route_stats_f.empty:
        route_stats_f = route_stats  # montrer quand même les routes

    top_routes = route_stats_f.head(top_k).reset_index()

    fig, ax = plt.subplots()
    sns.barplot(data=top_routes, y="Route", x="mean", orient="h", ax=ax)

    ax.set_xlabel("Retard moyen (minutes)")
    ax.set_ylabel("Liaison (départ → arrivée)")
    ax.set_title(f"Top {top_k} liaisons (n≥{actual_min_n})")

    for i, row in top_routes.iterrows():
        ax.text(
            float(row["mean"]), i, f"  n={int(row['count'])}", va="center", fontsize=8
        )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
