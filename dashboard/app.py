"""Streamlit dashboard: forecast of the coming Tempo days, season status, model quality.

Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, storage  # noqa: E402
from src.models import predict as P  # noqa: E402
from src.processing.features import Inputs  # noqa: E402
from src.processing.tempo_rules import season_label  # noqa: E402

st.set_page_config(page_title="Prévision Tempo EDF", layout="wide")

TEMPO_HEX = {"BLUE": "#2563eb", "WHITE": "#cbd5e1", "RED": "#dc2626"}
LABEL_FR = {"BLUE": "Bleu", "WHITE": "Blanc", "RED": "Rouge", None: "Inconnu"}
INK = "#334155"


@st.cache_data(ttl=300, show_spinner=False)
def load():
    storage.init_db()
    inputs = Inputs.from_db()
    try:
        fc = P.forecast(days=6, inputs=inputs)
    except Exception as exc:  # noqa: BLE001
        fc = {"error": str(exc)}
    try:
        meta = P.get_bundle().meta
    except P.ModelNotAvailable:
        meta = None
    history = P.prediction_history(horizon=1)
    return inputs, fc, meta, history


inputs, fc, meta, history = load()

st.title("Prévision des jours Tempo EDF")
st.caption(
    "Couleur officielle (RTE) quand elle est publiée, sinon prévision du modèle. "
    "Données : RTE éCO2mix (ODRÉ), api-couleur-tempo.fr, Open-Meteo."
)

if inputs.tempo.empty:
    st.error("Aucune donnée. Lancez le pipeline : `python -m src.pipeline run`")
    st.stop()
if "error" in fc:
    st.error(f"Erreur de prévision : {fc['error']}")
    st.stop()

# --- Next days cards ---------------------------------------------------------
st.subheader("Prochains jours")
cols = st.columns(len(fc["days"]))
for col, d in zip(cols, fc["days"]):
    color = d["color"]
    bg = TEMPO_HEX.get(color, "#f1f5f9")
    fg = "#0f172a" if color in ("WHITE", None) else "#ffffff"
    status = {"official": "officiel", "predicted": "prévu", "unavailable": "—"}[d["status"]]
    proba = ""
    if d["probabilities"]:
        p = d["probabilities"]
        proba = f"<div style='font-size:0.8em;margin-top:4px'>Bleu {p['BLUE']:.0%} · Blanc {p['WHITE']:.0%} · Rouge {p['RED']:.0%}</div>"
    col.markdown(
        f"<div style='background:{bg};color:{fg};border:1px solid #94a3b8;border-radius:10px;padding:12px;text-align:center'>"
        f"<div style='font-size:0.85em'>{d['weekday'].capitalize()} {pd.to_datetime(d['date']):%d/%m}</div>"
        f"<div style='font-size:1.5em;font-weight:700'>{LABEL_FR[color]}</div>"
        f"<div style='font-size:0.8em'>{status}</div>{proba}</div>",
        unsafe_allow_html=True,
    )

# --- Season status -----------------------------------------------------------
s = fc["season"]
st.subheader(f"Saison {s['season']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jours rouges restants", f"{s['red_left']} / {config.RED_QUOTA}")
c2.metric("Jours blancs restants", f"{s['white_left']} / {config.WHITE_QUOTA}")
c3.metric("Jours éligibles rouge restants", s["red_eligible_days_left"])
c4.metric("Dernière couleur officielle", pd.to_datetime(s["last_official_date"]).strftime("%d/%m/%Y"))

# --- Season calendar ---------------------------------------------------------
seasons = sorted(inputs.tempo["season"].unique(), reverse=True)
season = st.selectbox("Calendrier de la saison", seasons, index=0)
t = inputs.tempo[inputs.tempo["season"] == season].copy()
t["date"] = pd.to_datetime(t["date"])
start = pd.Timestamp(f"{season[:4]}-09-01")
days = pd.DataFrame({"date": pd.date_range(start, start + pd.offsets.DateOffset(years=1) - timedelta(days=1))})
days = days.merge(t[["date", "color"]], on="date", how="left")
if season == season_label(P.today_paris()):
    pred_map = {pd.Timestamp(d["date"]): d["color"] for d in fc["days"] if d["status"] == "predicted"}
    days["predicted"] = days["date"].map(pred_map)
else:
    days["predicted"] = None
days["week"] = ((days["date"] - (start - pd.Timedelta(days=start.weekday()))).dt.days // 7)
days["dow"] = days["date"].dt.dayofweek
code = {"BLUE": 0, "WHITE": 1, "RED": 2}
days["z"] = days["color"].map(code)
days["label"] = [
    f"{d:%a %d/%m/%Y} : {LABEL_FR.get(c) if isinstance(c, str) else ('prévu ' + LABEL_FR[p] if isinstance(p, str) else 'inconnu')}"
    for d, c, p in zip(days["date"], days["color"], days["predicted"])
]
fig = go.Figure(
    go.Heatmap(
        x=days["week"], y=days["dow"], z=days["z"], text=days["label"], hoverinfo="text",
        colorscale=[[0, TEMPO_HEX["BLUE"]], [0.33, TEMPO_HEX["BLUE"]], [0.34, TEMPO_HEX["WHITE"]],
                    [0.66, TEMPO_HEX["WHITE"]], [0.67, TEMPO_HEX["RED"]], [1, TEMPO_HEX["RED"]]],
        zmin=0, zmax=2, showscale=False, xgap=2, ygap=2,
    )
)
pred_days = days[days["predicted"].notna() & days["color"].isna()]
if len(pred_days):
    fig.add_trace(go.Scatter(
        x=pred_days["week"], y=pred_days["dow"], mode="markers", text=pred_days["label"], hoverinfo="text",
        marker=dict(symbol="square-open", size=12, color=[TEMPO_HEX[c] for c in pred_days["predicted"]], line=dict(width=2)),
        name="Prévu",
    ))
month_ticks = days.groupby(days["date"].dt.to_period("M"))["week"].min()
fig.update_layout(
    height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(tickvals=list(range(7)), ticktext=["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"], autorange="reversed"),
    xaxis=dict(tickvals=month_ticks.tolist(), ticktext=[p.strftime("%b") for p in month_ticks.index], showgrid=False),
)
st.plotly_chart(fig, width="stretch")
counts = t["color"].value_counts()
st.caption(" · ".join(f"{LABEL_FR[c]} : {counts.get(c, 0)}" for c in config.COLORS) + " — cases à contour : jours prévus")

# --- Consumption & temperature (two charts, one axis each) --------------------
st.subheader("Consommation et température (60 derniers jours + prévision météo)")
since = P.today_paris() - timedelta(days=60)
cons = inputs.consumption[(inputs.consumption["date"] >= since) & (inputs.consumption["coverage"] >= 0.9)]
weather = inputs.weather[inputs.weather["date"] >= since]
g1, g2 = st.columns(2)
with g1:
    f = go.Figure(go.Scatter(x=cons["date"], y=cons["conso_mean"] / 1000, mode="lines", line=dict(width=2, color="#2563eb"),
                             name="Consommation", hovertemplate="%{x|%d/%m} : %{y:.1f} GW<extra></extra>"))
    f.update_layout(title="Consommation moyenne journalière (GW)", height=320, hovermode="x unified",
                    margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(f, width="stretch")
with g2:
    obs = weather[weather["source"] == "archive"]
    fut = weather[weather["source"] == "forecast"]
    f = go.Figure()
    f.add_trace(go.Scatter(x=obs["date"], y=obs["temp_mean"], mode="lines", name="Observée", line=dict(width=2, color="#0f766e"),
                           hovertemplate="%{x|%d/%m} : %{y:.1f} °C<extra>observée</extra>"))
    f.add_trace(go.Scatter(x=fut["date"], y=fut["temp_mean"], mode="lines+markers", name="Prévue",
                           line=dict(width=2, dash="dot", color="#0f766e"), marker=dict(size=8),
                           hovertemplate="%{x|%d/%m} : %{y:.1f} °C<extra>prévue</extra>"))
    f.update_layout(title="Température France pondérée (°C)", height=320, hovermode="x unified",
                    margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(f, width="stretch")

# --- Model quality -------------------------------------------------------------
st.subheader("Qualité du modèle")
if meta is None:
    st.warning("Pas de modèle entraîné.")
else:
    st.caption(f"Modèle v{meta['version']} — entraîné le {pd.to_datetime(meta['trained_at']):%d/%m/%Y %H:%M} UTC. "
               "Backtest : chaque saison est prédite par un modèle entraîné uniquement sur les saisons précédentes.")
    rows = []
    for h, m in meta["horizons"].items():
        o = m.get("backtest", {}).get("overall")
        if not o:
            continue
        for name, key in (("Modèle", "model"), ("Heuristique température", "baseline_heuristic"), ("Toujours bleu", "baseline_always_blue")):
            sc = o[key]
            rows.append({"Horizon": f"J+{h}", "Méthode": name, "F1 macro": sc["f1_macro"], "Exactitude": sc["accuracy"],
                         "Rappel rouge": sc["recall_red"], "Précision rouge": sc["precision_red"], "Rappel blanc": sc["recall_white"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                    ["F1 macro", "Exactitude", "Rappel rouge", "Précision rouge", "Rappel blanc"]})
    h1 = meta["horizons"].get("1", {})
    cm = h1.get("backtest", {}).get("overall", {}).get("model", {}).get("confusion_matrix")
    if cm:
        with st.expander("Matrice de confusion (J+1, backtest)"):
            st.dataframe(pd.DataFrame(cm, index=[f"réel {LABEL_FR[c]}" for c in config.COLORS],
                                      columns=[f"prévu {LABEL_FR[c]}" for c in config.COLORS]))
    with st.expander("Importance des variables (J+1)"):
        imp = pd.Series(h1.get("feature_importance", {})).head(15).sort_values()
        f = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#2563eb",
                             hovertemplate="%{y} : %{x:.3f}<extra></extra>"))
        f.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(f, width="stretch")

st.subheader("Historique des prévisions réellement émises (J+1)")
done = history[history["official_color"].notna()]
if done.empty:
    st.info("Le suivi en conditions réelles se remplit au fil des exécutions quotidiennes du pipeline.")
else:
    st.metric("Exactitude en conditions réelles", f"{done['correct'].astype(float).mean():.0%}", help=f"{len(done)} jours évalués")
st.dataframe(history[["target_date", "pred_color", "official_color", "p_blue", "p_white", "p_red", "model_version"]]
             .sort_values("target_date", ascending=False), hide_index=True, width="stretch")
