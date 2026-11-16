"""Possession debug dashboard — frame-by-frame inspection of any labeled game.

Usage
-----
    streamlit run src/dashboard/app.py

    # Oct 20 BAR-ATH:
    DASHBOARD_GAME_DIR=data/tactical streamlit run src/dashboard/app.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.sample_loader import load_tracking, load_gt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_GAME_DIR = os.environ.get(
    "DASHBOARD_GAME_DIR",
    os.path.join(ROOT, "data", "sample", "Sample_Game_3"),
)
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
PITCH_L, PITCH_W = 105.0, 68.0

LABEL_COLOR = {"H": "#3b82f6", "A": "#ef4444", "D": "#9ca3af", "X": "#d1d5db"}
LABEL_NAME  = {"H": "Home", "A": "Away", "D": "Dead", "X": "Ungraded"}
TIMELINE_WINDOW = 250


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_game(game_dir: str):
    data = load_tracking(game_dir)
    gt   = load_gt(game_dir)
    return data, gt


@st.cache_data
def load_predictions(name: str) -> np.ndarray:
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    return df["label"].values


def available_game_dirs() -> list:
    """Return all game directories found under data/, sorted by name."""
    dirs = []
    data_root = os.path.join(ROOT, "data")
    if os.path.isdir(data_root):
        for parent, subdirs, _ in os.walk(data_root):
            subdirs.sort()
            for sub in subdirs:
                full = os.path.join(parent, sub)
                # A game dir contains a tracking CSV/TXT or events file
                files = os.listdir(full)
                if any(f.endswith(("_tracking_home.csv", "_RawTrackingData_Home_Team.csv",
                                    "_RawEventsData.csv", "_tracking.txt")) for f in files):
                    dirs.append(full)
    return dirs or [_DEFAULT_GAME_DIR]


def available_predictions() -> list:
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted(
        f[:-4] for f in os.listdir(RESULTS_DIR)
        if f.lower().endswith(".csv")
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def pitch_figure(home_xy, away_xy, ball_xy, frame: int,
                 pred_label: str, gt_label: str) -> go.Figure:
    """Plotly pitch with player dots and ball for one frame."""
    fig = go.Figure()

    # Pitch outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=PITCH_L, y1=PITCH_W,
                  line=dict(color="white", width=2))
    # Centre line
    fig.add_shape(type="line", x0=PITCH_L/2, y0=0, x1=PITCH_L/2, y1=PITCH_W,
                  line=dict(color="white", width=1, dash="dash"))
    # Centre circle (approximate)
    theta = np.linspace(0, 2*np.pi, 60)
    r = 9.15
    fig.add_trace(go.Scatter(
        x=PITCH_L/2 + r*np.cos(theta),
        y=PITCH_W/2 + r*np.sin(theta),
        mode="lines", line=dict(color="white", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    # Home players
    hx = home_xy[frame, :, 0] * PITCH_L
    hy = home_xy[frame, :, 1] * PITCH_W
    visible = ~np.isnan(hx)
    fig.add_trace(go.Scatter(
        x=hx[visible], y=hy[visible],
        mode="markers",
        marker=dict(color=LABEL_COLOR["H"], size=10, symbol="circle"),
        name="Home", hovertemplate="Home<br>x=%.1f y=%.1f<extra></extra>",
    ))

    # Away players
    ax = away_xy[frame, :, 0] * PITCH_L
    ay = away_xy[frame, :, 1] * PITCH_W
    visible = ~np.isnan(ax)
    fig.add_trace(go.Scatter(
        x=ax[visible], y=ay[visible],
        mode="markers",
        marker=dict(color=LABEL_COLOR["A"], size=10, symbol="circle"),
        name="Away", hovertemplate="Away<br>x=%.1f y=%.1f<extra></extra>",
    ))

    # Ball
    bx, by = ball_xy[frame]
    if not np.isnan(bx):
        fig.add_trace(go.Scatter(
            x=[bx * PITCH_L], y=[by * PITCH_W],
            mode="markers",
            marker=dict(color="#facc15", size=14, symbol="circle",
                        line=dict(color="black", width=1)),
            name="Ball",
        ))

    pred_color = LABEL_COLOR[pred_label]
    gt_color   = LABEL_COLOR[gt_label]
    match_icon = "✓" if pred_label == gt_label else "✗"

    fig.update_layout(
        title=dict(
            text=(f"Frame {frame} &nbsp;|&nbsp; "
                  f"<span style='color:{pred_color}'>Pred: {LABEL_NAME[pred_label]}</span>"
                  f" &nbsp;{match_icon}&nbsp; "
                  f"<span style='color:{gt_color}'>GT: {LABEL_NAME[gt_label]}</span>"),
            x=0.5,
        ),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#166534",
        font=dict(color="white"),
        xaxis=dict(range=[-2, PITCH_L+2], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(range=[-2, PITCH_W+2], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.02),
        height=420,
    )
    return fig


def timeline_figure(pred: np.ndarray, gt: np.ndarray,
                    frame: int) -> go.Figure:
    """Two horizontal bar tracks (pred / GT) over the last 250 frames."""
    lo = max(0, frame - TIMELINE_WINDOW + 1)
    hi = frame + 1
    frames = np.arange(lo, hi)

    fig = go.Figure()

    for track_y, labels, track_name in [(1, pred[lo:hi], "Pred"),
                                         (0, gt[lo:hi],   "GT")]:
        i = 0
        n = len(labels)
        while i < n:
            label = labels[i]
            j = i + 1
            while j < n and labels[j] == label:
                j += 1
            fig.add_trace(go.Bar(
                x=[j - i],
                y=[track_name],
                orientation="h",
                base=frames[i],
                marker_color=LABEL_COLOR[label],
                name=LABEL_NAME[label],
                showlegend=False,
                hovertemplate=f"{LABEL_NAME[label]} f{frames[i]}–{frames[j-1]}<extra></extra>",
            ))
            i = j

    # Current frame marker
    fig.add_vline(x=frame, line_color="white", line_width=2)

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(color="white"),
        xaxis=dict(range=[lo, hi], showgrid=False, zeroline=False,
                   title="frame"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=30, b=30),
        height=130,
        title=dict(text="Possession timeline (last 250 frames)", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Possession Debug", layout="wide",
                   page_icon="⚽")

# Sidebar
with st.sidebar:
    st.header("Settings")
    _game_dirs = available_game_dirs()
    _default_idx = next(
        (i for i, d in enumerate(_game_dirs) if d == _DEFAULT_GAME_DIR), 0
    )
    game_dir = st.selectbox(
        "Game directory",
        _game_dirs,
        index=_default_idx,
        format_func=os.path.basename,
    )
    _preds = available_predictions()
    if not _preds:
        st.warning("No prediction CSVs found in `evaluation/results/`. Run `task train` first.")
        st.stop()
    pred_name = st.selectbox("Prediction file", _preds,
                              index=_preds.index("lstm_game3")
                              if "lstm_game3" in _preds else 0)
    st.caption(f"`evaluation/results/{pred_name}.csv`")

st.title(f"⚽ Possession Debug Dashboard — {os.path.basename(game_dir)}")
data, gt = load_game(game_dir)
pred = load_predictions(pred_name)

N = len(gt)
n_pred = len(pred)
if n_pred != N:
    st.warning(f"Prediction has {n_pred} rows but GT has {N}. Truncating to min.")
    n = min(N, n_pred)
    gt   = gt[:n]
    pred = pred[:n]
    N    = n

with st.sidebar:
    frame = st.slider("Frame", 0, N - 1, 0, step=1)
    st.markdown("---")
    st.metric("Pred", f"{LABEL_NAME[pred[frame]]} ({pred[frame]})")
    st.metric("GT",   f"{LABEL_NAME[gt[frame]]} ({gt[frame]})")
    correct = int((pred == gt).sum())
    st.metric("Frame accuracy", f"{correct/N:.1%}")

# Main panels
col_pitch, col_info = st.columns([3, 1])

with col_pitch:
    st.plotly_chart(
        pitch_figure(data["home_xy"], data["away_xy"], data["ball_xy"],
                     frame, pred[frame], gt[frame]),
        width="stretch",
    )

with col_info:
    st.subheader("Frame info")
    st.write(f"**Period:** {data['periods'][frame]}")
    bx, by = data["ball_xy"][frame]
    if np.isnan(bx):
        st.write("**Ball:** missing")
    else:
        st.write(f"**Ball:** ({bx*PITCH_L:.1f}, {by*PITCH_W:.1f}) m")

    n_home = int((~np.isnan(data["home_xy"][frame, :, 0])).sum())
    n_away = int((~np.isnan(data["away_xy"][frame, :, 0])).sum())
    st.write(f"**Home visible:** {n_home}")
    st.write(f"**Away visible:** {n_away}")

    st.subheader("Label legend")
    for label, name in LABEL_NAME.items():
        color = LABEL_COLOR[label]
        st.markdown(
            f"<span style='background:{color};padding:2px 8px;"
            f"border-radius:3px;color:white'>{label}</span> {name}",
            unsafe_allow_html=True,
        )

st.plotly_chart(
    timeline_figure(pred, gt, frame),
    width="stretch",
)
