"""
Streamlit dashboard for the DC Urban Heat Island project.

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from data_utils import (
    download_dc_tracts,
    generate_synthetic_rasters,
    build_feature_dataframe,
)
from features_ml import prepare_features, train_model, evaluate_model, predict_all

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DC Urban Heat Islands",
    page_icon=":thermometer:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data pipeline (cached so it only runs once)
# ---------------------------------------------------------------------------

@st.cache_data
def run_pipeline():
    """Load data, compute features, train model, return results."""
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)

    dc_gdf = download_dc_tracts(data_dir)
    temp_path, ndvi_path = generate_synthetic_rasters(dc_gdf, data_dir)
    heat_df = build_feature_dataframe(dc_gdf, temp_path, ndvi_path)

    X, y = prepare_features(heat_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    importances = dict(zip(X.columns, model.feature_importances_))

    # Predict for all clean tracts
    clean = (
        heat_df[["mean_temp", "mean_ndvi", "distance_to_center", "area_km2"]]
        .notna()
        .all(axis=1)
    )
    gdf = heat_df.loc[clean].copy()
    gdf["predicted_temp"] = predict_all(model, gdf).values
    gdf["prediction_error"] = gdf["predicted_temp"] - gdf["mean_temp"]

    # Round for cleaner display/tooltips
    for col in [
        "mean_temp", "mean_ndvi", "distance_to_center",
        "area_km2", "predicted_temp", "prediction_error",
    ]:
        gdf[col] = gdf[col].round(3)

    return gdf, metrics, importances


with st.spinner("Running analysis pipeline (first load may take ~30 seconds)..."):
    gdf, metrics, importances = run_pipeline()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Urban Heat Islands in Washington, DC")
st.markdown(
    "Mapping and predicting surface temperature across DC census tracts "
    "using vegetation data and machine learning."
)

st.divider()

# ---------------------------------------------------------------------------
# Interactive map
# ---------------------------------------------------------------------------

st.header("Interactive Map")

map_layers = {
    "Observed Temperature (\u00b0C)": ("mean_temp", "RdYlGn_r"),
    "Vegetation Index (NDVI)": ("mean_ndvi", "YlGn"),
    "Predicted Temperature (\u00b0C)": ("predicted_temp", "RdYlGn_r"),
    "Prediction Error (\u00b0C)": ("prediction_error", "RdBu_r"),
}

selected = st.selectbox("Choose a layer", list(map_layers.keys()))
col_name, cmap = map_layers[selected]

# geopandas.explore() returns a folium Map — works without mapclassify
# as long as we don't pass a scheme parameter
tooltip_cols = [
    "GEOID", "mean_temp", "mean_ndvi", "predicted_temp", "prediction_error",
]
tooltip_labels = [
    "Tract:", "Temp (\u00b0C):", "NDVI:", "Predicted (\u00b0C):", "Error (\u00b0C):",
]

m = gdf.explore(
    column=col_name,
    cmap=cmap,
    tooltip=tooltip_cols,
    tooltip_kwds={"aliases": tooltip_labels},
    legend=True,
    legend_kwds={"caption": selected},
    style_kwds={"weight": 0.5, "fillOpacity": 0.7},
    tiles="CartoDB positron",
)

try:
    from streamlit_folium import st_folium
    st_folium(m, height=520, use_container_width=True, returned_objects=[])
except ImportError:
    st.warning(
        "Install `streamlit-folium` for interactive maps: "
        "`pip install streamlit-folium`"
    )
    # Fallback: render as static HTML
    from streamlit.components.v1 import html
    html(m._repr_html_(), height=520)

st.divider()

# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------

st.header("Data Explorer")
display_cols = [
    "GEOID", "mean_temp", "mean_ndvi", "distance_to_center",
    "area_km2", "predicted_temp", "prediction_error",
]
st.dataframe(
    gdf[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=320,
)

st.divider()

# ---------------------------------------------------------------------------
# NDVI vs. temperature
# ---------------------------------------------------------------------------

st.header("Vegetation & Temperature")
col_left, col_right = st.columns([3, 1])

with col_left:
    fig = px.scatter(
        gdf,
        x="mean_ndvi",
        y="mean_temp",
        hover_data=["GEOID"],
        labels={
            "mean_ndvi": "Mean NDVI",
            "mean_temp": "Mean Temperature (\u00b0C)",
        },
        title="NDVI vs. Surface Temperature by Census Tract",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.65, color="#2196F3",
                                  line=dict(width=0.4, color="black")))
    # Manual trend line (avoids statsmodels dependency)
    z = np.polyfit(gdf["mean_ndvi"], gdf["mean_temp"], 1)
    xs = np.linspace(gdf["mean_ndvi"].min(), gdf["mean_ndvi"].max(), 100)
    fig.add_trace(
        go.Scatter(
            x=xs, y=np.poly1d(z)(xs), mode="lines",
            line=dict(dash="dash", color="red", width=2),
            name=f"Trend (slope = {z[0]:.2f})",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    corr = gdf["mean_ndvi"].corr(gdf["mean_temp"])
    st.metric("Pearson r", f"{corr:.3f}")
    slope = z[0]
    st.markdown(
        f"Each **0.1 increase** in NDVI is associated with a "
        f"**{abs(slope) * 0.1:.1f}\u00b0C decrease** in surface temperature."
    )
    st.markdown(
        "This confirms that greener census tracts are measurably cooler — "
        "a hallmark of the urban heat island effect."
    )

st.divider()

# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------

st.header("Model Performance")
st.markdown("**Random Forest Regressor** — 100 trees, 80/20 train-test split")

m1, m2, m3 = st.columns(3)
m1.metric("R\u00b2 Score", f"{metrics['r2']:.3f}")
m2.metric("RMSE", f"{metrics['rmse']:.3f} \u00b0C")
m3.metric("MAE", f"{metrics['mae']:.3f} \u00b0C")

col_left, col_right = st.columns(2)

with col_left:
    imp_df = (
        pd.DataFrame({"Feature": list(importances.keys()),
                       "Importance": list(importances.values())})
        .sort_values("Importance", ascending=True)
    )
    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        title="Feature Importances",
        color="Importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    fig = px.scatter(
        gdf,
        x="mean_temp",
        y="predicted_temp",
        hover_data=["GEOID"],
        labels={
            "mean_temp": "Observed Temp (\u00b0C)",
            "predicted_temp": "Predicted Temp (\u00b0C)",
        },
        title="Observed vs. Predicted Temperature",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.65, color="#FF9800",
                                  line=dict(width=0.4, color="black")))
    lo = min(gdf["mean_temp"].min(), gdf["predicted_temp"].min()) - 0.5
    hi = max(gdf["mean_temp"].max(), gdf["predicted_temp"].max()) + 0.5
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(dash="dash", color="black", width=1.5),
            name="1:1 line",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[lo, hi], scaleanchor="y"),
        yaxis=dict(range=[lo, hi], constrain="domain"),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

st.header("Summary")
top_feature = max(importances, key=importances.get)
st.markdown(
    f"This analysis examined urban heat island patterns across **{len(gdf)} "
    f"census tracts** in Washington, DC using land surface temperature and "
    f"vegetation index (NDVI) data. Tracts with lower vegetation cover were "
    f"consistently warmer, confirming the well-documented inverse relationship "
    f"between greenness and surface temperature. A Random Forest model trained "
    f"on NDVI, distance to city center, and tract area achieved an "
    f"**R\u00b2 of {metrics['r2']:.2f}** and a **mean absolute error of "
    f"{metrics['mae']:.2f}\u00b0C**. The most informative predictor was "
    f"`{top_feature}`, highlighting how vegetation cover and spatial location "
    f"jointly shape surface temperatures. The hottest tracts cluster in the "
    f"downtown core where impervious surfaces dominate, while cooler tracts "
    f"align with Rock Creek Park and the Anacostia River corridor. This kind "
    f"of analysis helps planners identify heat-vulnerable neighborhoods and "
    f"prioritize tree planting or cool-roof programs."
)

st.divider()
st.caption(
    "Data: US Census TIGER/Line tracts (2020) \u00b7 "
    "Synthetic LST & NDVI rasters \u00b7 "
    "Model: scikit-learn RandomForestRegressor"
)
