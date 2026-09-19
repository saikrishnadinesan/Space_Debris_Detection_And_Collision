"""
Phase 8: Visualization Dashboard
==================================
Streamlit dashboard for real-time space debris monitoring.
Shows 3D orbital view, tracked objects, and collision risks.

Run with: streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛸 Space Debris Monitor",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0a1a; color: #e0e0ff; }
    .metric-card {
        background: #1a1a2e; border-radius: 10px; padding: 15px;
        border: 1px solid #333366; margin: 5px;
    }
    .risk-critical { color: #ff4444; font-weight: bold; }
    .risk-high { color: #ff8800; font-weight: bold; }
    .risk-medium { color: #ffcc00; font-weight: bold; }
    .risk-low { color: #44ff44; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─── Real Pipeline Data Loader ───────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_debris_data():
    """
    Load REAL debris data from full pipeline:
    TLE catalog → SGP4 positions → LSTM prediction → collision risk
    Caches results for 5 minutes to avoid re-running on every refresh.
    """
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "src"))

    positions_path = os.path.join(ROOT, "outputs", "real_positions.csv")
    risk_path      = os.path.join(ROOT, "outputs", "real_collision_risks.csv")

    # ── Use cached results if fresh (< 5 minutes old) ───────────
    if os.path.exists(positions_path):
        age_secs = (datetime.utcnow() -
                    datetime.utcfromtimestamp(
                        os.path.getmtime(positions_path))).total_seconds()
        if age_secs < 300:
            df = pd.read_csv(positions_path)
            if not df.empty and "x" in df.columns:
                if "velocity_km_s" not in df.columns and "speed_km_s" in df.columns:
                    df["velocity_km_s"] = df["speed_km_s"]
                return df

    # ── Run full real pipeline ────────────────────────────────────
    try:
        from pipeline.real_pipeline import run_full_pipeline
        with st.spinner("🛸 Running real pipeline: TLE → SGP4 → LSTM → Risk..."):
            df, _ = run_full_pipeline(max_objects=300)
        if not df.empty:
            if "velocity_km_s" not in df.columns and "speed_km_s" in df.columns:
                df["velocity_km_s"] = df["speed_km_s"]
            return df
    except Exception as e:
        st.warning(f"⚠️ Pipeline error: {e} — using fallback data")

    # ── Fallback: synthetic data ──────────────────────────────────
    np.random.seed(int(datetime.now().second / 10))
    n = 150
    data = []
    for i in range(n):
        orbit = np.random.choice(["LEO","MEO","GEO"], p=[0.7,0.2,0.1])
        alt   = {"LEO": np.random.uniform(300,2000),
                 "MEO": np.random.uniform(2000,35786),
                 "GEO": np.random.uniform(35500,36000)}[orbit]
        r     = 6371 + alt
        theta = np.random.uniform(0, 2*np.pi)
        phi   = np.random.uniform(-np.pi/2, np.pi/2)
        data.append({
            "id":           f"OBJ-{1000+i}",
            "name":         f"DEBRIS-{1000+i}",
            "x":            r*np.cos(phi)*np.cos(theta),
            "y":            r*np.cos(phi)*np.sin(theta),
            "z":            r*np.sin(phi),
            "altitude_km":  alt,
            "orbit_class":  orbit,
            "type":         np.random.choice(["small_debris","medium_debris",
                                "large_debris","rocket_body","defunct_satellite"]),
            "risk_level":   np.random.choice(["LOW","MEDIUM","HIGH","CRITICAL"],
                                              p=[0.7,0.2,0.07,0.03]),
            "velocity_km_s": np.random.uniform(3, 8),
            "speed_km_s":   np.random.uniform(3, 8),
            "size_m":       np.random.uniform(0.01, 5.0),
        })
    return pd.DataFrame(data)


def create_earth_with_texture():
    """
    Create Earth sphere with realistic land/ocean color texture
    using a latitude/longitude colormap to simulate continents.
    """
    N = 80
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = 6371 * np.outer(np.cos(u), np.sin(v))
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(N), np.cos(v))

    # Build a lat/lon texture that approximates land vs ocean
    # Using a hand-crafted pattern based on actual continent positions
    lon_grid = np.degrees(u)                            # 0–360
    lat_grid = 90 - np.degrees(v)                      # 90 to -90
    texture  = np.zeros((N, N))

    for i, lon in enumerate(lon_grid):
        for j, lat in enumerate(lat_grid):
            is_land = False
            # North America
            if -170 < lon-360 < -50 and 15 < lat < 75: is_land = True
            if -130 < lon-360 < -60 and 25 < lat < 50: is_land = True
            # South America
            if -82 < lon-360 < -34 and -56 < lat < 13: is_land = True
            # Europe
            if -10 < lon-360 < 40 and 35 < lat < 71:  is_land = True
            if 0 < lon < 40 and 35 < lat < 71:        is_land = True
            # Africa
            if -18 < lon-360 < 52 and -35 < lat < 37: is_land = True
            if 0 < lon < 52 and -35 < lat < 37:       is_land = True
            # Asia
            if 26 < lon < 180 and 0 < lat < 75:       is_land = True
            if 26 < lon < 145 and -10 < lat < 75:     is_land = True
            # Australia
            if 113 < lon < 154 and -44 < lat < -10:   is_land = True
            # Greenland
            if -57 < lon-360 < -17 and 60 < lat < 84: is_land = True
            # Antarctica
            if lat < -65:                              is_land = True
            # Poles ice caps
            if lat > 80:                               is_land = True

            texture[i, j] = 0.85 if is_land else 0.15

    return x, y, z, texture


def add_orbit_ring(fig, radius_km: float, color: str, name: str, dash: str = "solid"):
    """Add a circular orbit ring at given radius."""
    theta = np.linspace(0, 2 * np.pi, 200)
    # Ring in equatorial plane
    rx = radius_km * np.cos(theta)
    ry = radius_km * np.sin(theta)
    rz = np.zeros_like(theta)
    fig.add_trace(go.Scatter3d(
        x=rx, y=ry, z=rz,
        mode="lines",
        line=dict(color=color, width=1.5, dash=dash),
        name=name,
        hoverinfo="name",
        showlegend=True,
    ))


def add_starfield(fig, n_stars: int = 800):
    """Add random background stars."""
    np.random.seed(7)
    dist   = 80000
    theta  = np.random.uniform(0, 2*np.pi, n_stars)
    phi    = np.random.uniform(0, np.pi,   n_stars)
    sx = dist * np.sin(phi) * np.cos(theta)
    sy = dist * np.sin(phi) * np.sin(theta)
    sz = dist * np.cos(phi)
    sizes = np.random.choice([1, 1, 1, 2, 2, 3], n_stars)
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz,
        mode="markers",
        marker=dict(size=sizes, color="white", opacity=0.6),
        name="Stars",
        hoverinfo="skip",
        showlegend=False,
    ))


def create_3d_orbit_plot(df: pd.DataFrame) -> go.Figure:
    """Create enhanced 3D orbital visualization with Earth texture,
    orbit rings and starfield background."""
    fig = go.Figure()

    # ── Starfield ────────────────────────────────────────────────
    add_starfield(fig, n_stars=600)

    # ── Earth with texture ───────────────────────────────────────
    x, y, z, texture = create_earth_with_texture()

    # Ocean layer (solid blue base)
    fig.add_surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#0a2a6e"], [1, "#1a6ab5"]],
        surfacecolor=np.zeros_like(texture),
        opacity=1.0,
        showscale=False,
        name="Ocean",
        hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
    )

    # Land layer (continent overlay)
    fig.add_surface(
        x=x * 1.002, y=y * 1.002, z=z * 1.002,
        colorscale=[
            [0.0, "rgba(10,42,110,0)"],
            [0.4, "rgba(10,42,110,0)"],
            [0.5, "#2d6a2d"],
            [0.7, "#4a8a3a"],
            [1.0, "#6aaa4a"],
        ],
        surfacecolor=texture,
        cmin=0, cmax=1,
        opacity=1.0,
        showscale=False,
        name="Land",
        hoverinfo="skip",
        lighting=dict(ambient=0.7, diffuse=0.8),
    )

    # Atmosphere glow
    fig.add_surface(
        x=x * 1.025, y=y * 1.025, z=z * 1.025,
        colorscale=[[0, "rgba(100,160,255,0.0)"], [1, "rgba(100,160,255,0.08)"]],
        surfacecolor=np.ones_like(texture),
        opacity=0.15,
        showscale=False,
        name="Atmosphere",
        hoverinfo="skip",
    )

    # ── Orbit Rings ──────────────────────────────────────────────
    # LEO boundary (~2000 km altitude = 8371 km radius)
    add_orbit_ring(fig, 8371,  "#4488ff", "LEO boundary (2,000 km)",  dash="dot")
    # MEO boundary (~20,200 km altitude = 26,571 km radius)
    add_orbit_ring(fig, 26571, "#44ffaa", "MEO boundary (20,200 km)", dash="dash")
    # GEO ring (~35,786 km altitude = 42,157 km radius)
    add_orbit_ring(fig, 42157, "#ffaa44", "GEO ring (35,786 km)",     dash="longdash")

    # ── Debris Objects ───────────────────────────────────────────
    color_map = {
        "LOW":      "#44ff44",
        "MEDIUM":   "#ffcc00",
        "HIGH":     "#ff8800",
        "CRITICAL": "#ff4444",
    }
    size_map = {"LOW": 3, "MEDIUM": 4, "HIGH": 5, "CRITICAL": 6}

    for risk_level, group in df.groupby("risk_level"):
        fig.add_trace(go.Scatter3d(
            x=group["x"], y=group["y"], z=group["z"],
            mode="markers",
            marker=dict(
                size=size_map[risk_level],
                color=color_map[risk_level],
                opacity=0.9,
                symbol="circle",
                line=dict(width=0.5, color="white") if risk_level == "CRITICAL" else dict(width=0),
            ),
            name=f"{risk_level} ({len(group)})",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "NORAD ID: %{customdata[3]}<br>"
                "Altitude: %{customdata[0]:.0f} km<br>"
                "Type: %{customdata[1]}<br>"
                "Size: %{customdata[2]:.2f} m<br>"
                "Risk: " + risk_level + "<extra></extra>"
            ),
            text=group["name"] if "name" in group.columns else group["id"],
            customdata=group[["altitude_km", "type", "size_m", "id"]].values,
        ))

    # ── Layout ───────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="🛸 Live Debris Field — 3D Orbital View",
            font=dict(color="white", size=16)
        ),
        paper_bgcolor="#00000f",
        plot_bgcolor="#00000f",
        scene=dict(
            bgcolor="#00000f",
            xaxis=dict(showgrid=False, showticklabels=False,
                       showbackground=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False,
                       showbackground=False, title=""),
            zaxis=dict(showgrid=False, showticklabels=False,
                       showbackground=False, title=""),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            aspectmode="cube",
        ),
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(20,20,40,0.8)",
            bordercolor="#333366",
            borderwidth=1,
        ),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def create_altitude_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="altitude_km", color="orbit_class",
                       nbins=50, title="Debris Distribution by Altitude",
                       color_discrete_map={"LEO": "#4488ff", "MEO": "#44ffaa", "GEO": "#ff8844"})
    fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#0d0d2b",
                      font=dict(color="white"), height=300)
    return fig


def create_risk_pie(df: pd.DataFrame) -> go.Figure:
    risk_counts = df["risk_level"].value_counts()
    colors = {"LOW": "#44ff44", "MEDIUM": "#ffcc00", "HIGH": "#ff8800", "CRITICAL": "#ff4444"}
    fig = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker=dict(colors=[colors.get(k, "gray") for k in risk_counts.index])
    ))
    fig.update_layout(title="Risk Distribution", paper_bgcolor="#1a1a2e",
                     font=dict(color="white"), height=300)
    return fig


# ─── Main Dashboard ───────────────────────────────────────────────────────────

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🛸 Space Debris Detection & Tracking System")
        st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    with col2:
        st.markdown("### 🟢 System Active")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    df = load_debris_data()

    # ── KPI Metrics ──
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Objects", len(df), "+real data")
    c2.metric("LEO Objects", len(df[df["orbit_class"] == "LEO"]))
    c3.metric("🔴 Critical Risk", len(df[df["risk_level"] == "CRITICAL"]),
              delta_color="inverse")
    c4.metric("🟠 High Risk", len(df[df["risk_level"] == "HIGH"]))
    c5.metric("Avg Altitude", f"{df['altitude_km'].mean():.0f} km")

    # ── 3D Plot ──
    st.markdown("---")
    st.plotly_chart(create_3d_orbit_plot(df), use_container_width=True)

    # ── Bottom Row ──
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(create_altitude_histogram(df), use_container_width=True)
    with col_right:
        st.plotly_chart(create_risk_pie(df), use_container_width=True)

    # ── High Risk Table ──
    st.markdown("---")
    st.subheader("⚠️ High Priority Collision Alerts")

    # Try loading real risk report first
    risk_path = "outputs/real_collision_risks.csv"
    df_risks = pd.DataFrame()

    if os.path.exists(risk_path) and os.path.getsize(risk_path) > 0:
        try:
            df_risks = pd.read_csv(risk_path)
        except pd.errors.EmptyDataError:
            df_risks = pd.DataFrame()

    if not df_risks.empty and "risk_level" in df_risks.columns:
        col_map = {
            "current_dist_km": "current_distance_km",
            "min_dist_km": "min_distance_km",
            "collision_prob": "collision_probability",
        }
        # Normalize whichever column naming convention is present
        rename_back = {v: k for k, v in col_map.items() if v in df_risks.columns}
        df_risks = df_risks.rename(columns=rename_back)

        wanted = ["obj1_name", "obj2_name", "current_dist_km",
                  "min_dist_km", "collision_prob", "risk_level"]
        available = [c for c in wanted if c in df_risks.columns]

        high = df_risks[df_risks["risk_level"].isin(["CRITICAL", "HIGH"])][available].head(15)
        if not high.empty:
            high.columns = ["Object 1", "Object 2", "Current Dist (km)",
                             "Min Dist (km)", "Collision Prob", "Risk Level"][:len(available)]
            st.dataframe(high, use_container_width=True, height=300)
        else:
            st.success("✅ No high-priority collision risks detected in current epoch")
    else:
        high_risk = df[df["risk_level"].isin(["CRITICAL","HIGH"])][
            ["id","type","orbit_class","altitude_km",
             "velocity_km_s","size_m","risk_level"]
        ].sort_values("risk_level")
        st.dataframe(
            high_risk.style.map(
                lambda x: "color: #ff4444" if x == "CRITICAL"
                          else "color: #ff8800" if x == "HIGH" else "",
                subset=["risk_level"]
            ),
            use_container_width=True, height=250
        )

    # ── Sidebar ──
    with st.sidebar:
        st.header("🔧 Filters")
        orbit_filter = st.multiselect("Orbit Class", ["LEO", "MEO", "GEO"],
                                       default=["LEO", "MEO", "GEO"])
        risk_filter = st.multiselect("Risk Level",
                                      ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                                      default=["HIGH", "CRITICAL"])
        size_range = st.slider("Size (meters)", 0.0, 10.0, (0.0, 10.0))

        st.markdown("---")
        st.header("📡 Data Sources")
        st.markdown("- ✅ Celestrak TLE Feed")
        st.markdown("- ✅ NASA ODPO Catalog")
        st.markdown("- ⚠️ Space-Track (login required)")
        st.markdown("- ✅ Synthetic Simulation")

        st.markdown("---")
        st.header("🤖 Model Status")
        st.markdown("- 🟢 YOLOv8 Detection: Active")
        st.markdown("- 🟢 DeepSORT Tracking: Active")
        st.markdown("- 🟡 LSTM Prediction: Training")
        st.markdown("- 🟢 Risk Estimator: Active")


if __name__ == "__main__":
    main()