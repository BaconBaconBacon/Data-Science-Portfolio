"""
Visualization functions for the wildfire risk ML project.

Includes:
- Wildfire locations map (Folium)
- Property risk map (Folium)
- Feature importance bar chart (matplotlib)
- Actual vs predicted scatter plot (matplotlib)
"""

import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from load_census import census_code_to_label
from settings import HEADER_GEOM, FIGURES_DIR


def create_wildfire_map(
    wildfire_gdf: gpd.GeoDataFrame,
    center: Optional[tuple[float, float]] = None,
    zoom_start: int = 5,
    radius: int = 3,
    color: str = "red",
    save_path: Optional[Path] = None,
) -> folium.Map:
    """
    Create a Folium map showing wildfire locations.

    Uses FastMarkerCluster for performance — points are clustered at low
    zoom levels and expand as you zoom in.

    Parameters
    ----------
    wildfire_gdf : gpd.GeoDataFrame
        Wildfire data with LATITUDE and LONGITUDE columns.
    center : tuple, optional
        Map center as (lat, lon). Defaults to mean of wildfire locations.
    zoom_start : int
        Initial zoom level.
    radius : int
        Circle marker radius in pixels (unused with FastMarkerCluster).
    color : str
        Marker color (unused with FastMarkerCluster).
    save_path : Path, optional
        If provided, save the map to this HTML file.

    Returns
    -------
    folium.Map
        The generated map object.
    """
    from folium.plugins import FastMarkerCluster

    # Convert to WGS84 if needed
    if wildfire_gdf.crs and wildfire_gdf.crs.to_epsg() != 4326:
        wildfire_gdf = wildfire_gdf.to_crs(4326)

    # Default center
    if center is None:
        center = (wildfire_gdf["LATITUDE"].mean(), wildfire_gdf["LONGITUDE"].mean())

    m = folium.Map(location=center, zoom_start=zoom_start)

    # Use FastMarkerCluster for performance (all points, clustered on zoom)
    coords = wildfire_gdf[["LATITUDE", "LONGITUDE"]].values.tolist()
    print(f"Adding {len(coords)} wildfire points to map...")
    FastMarkerCluster(data=coords).add_to(m)

    # Add legend
    legend_html = """
        <div style="position: fixed;
             bottom: 50px; left: 50px; width: 150px; height: 60px;
             border:2px solid grey; z-index:9999; font-size:14px;
             background-color:white; opacity: 0.85; padding: 10px;">
             <b>Legend</b><br>
             <i class="fa fa-circle" style="color:red"></i> Wildfire
        </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        print(f"Map saved to {save_path}")

    return m


def create_property_risk_map(
    properties_gdf: gpd.GeoDataFrame,
    risk_column: str = "nearest_fire_km",
    center: Optional[tuple[float, float]] = None,
    zoom_start: int = 5,
    radius: int = 6,
    colormap: str = "RdYlGn",
    reverse_colors: bool = True,
    save_path: Optional[Path] = None,
) -> folium.Map:
    """
    Create a Folium map showing properties colored by risk score.

    Parameters
    ----------
    properties_gdf : gpd.GeoDataFrame
        Property data with geometry and a risk column.
    risk_column : str
        Column name containing the risk score.
    center : tuple, optional
        Map center as (lat, lon). Defaults to mean of property locations.
    zoom_start : int
        Initial zoom level.
    radius : int
        Circle marker radius in pixels.
    colormap : str
        Matplotlib colormap name.
    reverse_colors : bool
        If True, reverse colormap (useful when higher values = lower risk).
    save_path : Path, optional
        If provided, save the map to this HTML file.

    Returns
    -------
    folium.Map
        The generated map object.
    """
    # Convert to WGS84 if needed
    gdf = properties_gdf.copy()
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Extract lat/lon from geometry
    gdf["_lat"] = gdf.geometry.y
    gdf["_lon"] = gdf.geometry.x

    # Default center
    if center is None:
        center = (gdf["_lat"].mean(), gdf["_lon"].mean())

    m = folium.Map(location=center, zoom_start=zoom_start)

    # Normalize risk values for coloring
    risk_values = gdf[risk_column].values
    vmin, vmax = np.nanmin(risk_values), np.nanmax(risk_values)

    # Get colormap
    cmap = plt.cm.get_cmap(colormap)
    if reverse_colors:
        cmap = cmap.reversed()

    # Add property points
    for _, row in gdf.iterrows():
        risk = row[risk_column]
        if pd.isna(risk):
            color = "gray"
        else:
            # Normalize to 0-1
            norm_val = (risk - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            rgba = cmap(norm_val)
            color = (
                f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
            )

        folium.CircleMarker(
            location=[row["_lat"], row["_lon"]],
            radius=radius,
            fill=True,
            fill_opacity=0.8,
            weight=1,
            fill_color=color,
            color="black",
            popup=f"{risk_column}: {risk:.2f}" if not pd.isna(risk) else "N/A",
        ).add_to(m)

    # Add legend
    legend_html = f"""
        <div style="position: fixed;
             bottom: 50px; left: 50px; width: 180px; height: 100px;
             border:2px solid grey; z-index:9999; font-size:12px;
             background-color:white; opacity: 0.9; padding: 10px;">
             <b>{risk_column}</b><br>
             <span style="color:{'green' if reverse_colors else 'red'}">● High risk ({vmin:.1f})</span><br>
             <span style="color:yellow">● Medium risk</span><br>
             <span style="color:{'red' if reverse_colors else 'green'}">● Low risk ({vmax:.1f})</span>
        </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        print(f"Map saved to {save_path}")

    return m


def create_combined_map(
    properties_gdf: gpd.GeoDataFrame,
    wildfire_gdf: gpd.GeoDataFrame,
    risk_column: str = "nearest_fire_km",
    center: Optional[tuple[float, float]] = None,
    zoom_start: int = 5,
    save_path: Optional[Path] = None,
) -> folium.Map:
    """
    Create a Folium map showing both properties and wildfires.

    Uses FastMarkerCluster for wildfires (performance) and CircleMarkers
    for properties (to show risk coloring).

    Parameters
    ----------
    properties_gdf : gpd.GeoDataFrame
        Property data with geometry and risk column.
    wildfire_gdf : gpd.GeoDataFrame
        Wildfire data with LATITUDE and LONGITUDE columns.
    risk_column : str
        Column name containing the risk score.
    center : tuple, optional
        Map center as (lat, lon).
    zoom_start : int
        Initial zoom level.
    save_path : Path, optional
        If provided, save the map to this HTML file.

    Returns
    -------
    folium.Map
        The generated map object.
    """
    from folium.plugins import FastMarkerCluster

    # Convert to WGS84 if needed
    props = properties_gdf.copy()
    fires = wildfire_gdf.copy()

    if props.crs and props.crs.to_epsg() != 4326:
        props = props.to_crs(4326)
    if fires.crs and fires.crs.to_epsg() != 4326:
        fires = fires.to_crs(4326)

    props["_lat"] = props.geometry.y
    props["_lon"] = props.geometry.x

    # Default center
    if center is None:
        all_lats = list(props["_lat"]) + list(fires["LATITUDE"])
        all_lons = list(props["_lon"]) + list(fires["LONGITUDE"])
        center = (np.mean(all_lats), np.mean(all_lons))

    m = folium.Map(location=center, zoom_start=zoom_start)

    # Create feature groups for layer control
    fire_group = folium.FeatureGroup(name="Wildfires")
    prop_group = folium.FeatureGroup(name="Properties")

    # Add wildfire points using FastMarkerCluster
    print(f"Adding {len(fires)} wildfire points...")
    fire_coords = fires[["LATITUDE", "LONGITUDE"]].values.tolist()
    FastMarkerCluster(data=fire_coords).add_to(fire_group)

    # Add property points with risk coloring
    # Properties are typically fewer, so CircleMarkers are fine
    print(f"Adding {len(props)} property points...")
    if risk_column in props.columns:
        risk_values = props[risk_column].values
        vmin, vmax = np.nanmin(risk_values), np.nanmax(risk_values)
        cmap = plt.cm.get_cmap("RdYlGn").reversed()

        # Vectorized color computation
        norm_vals = (
            (risk_values - vmin) / (vmax - vmin)
            if vmax > vmin
            else np.full_like(risk_values, 0.5)
        )
        colors = [cmap(nv) for nv in norm_vals]
        hex_colors = [
            f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            for c in colors
        ]

        for i, (_, row) in enumerate(props.iterrows()):
            folium.CircleMarker(
                location=[row["_lat"], row["_lon"]],
                radius=6,
                fill=True,
                fill_opacity=0.9,
                weight=2,
                fill_color=hex_colors[i],
                color="black",
                popup=f"{risk_column}: {risk_values[i]:.2f}",
            ).add_to(prop_group)
    else:
        for _, row in props.iterrows():
            folium.CircleMarker(
                location=[row["_lat"], row["_lon"]],
                radius=6,
                fill=True,
                fill_opacity=0.9,
                weight=2,
                fill_color="blue",
                color="black",
            ).add_to(prop_group)

    fire_group.add_to(m)
    prop_group.add_to(m)
    folium.LayerControl().add_to(m)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        print(f"Map saved to {save_path}")

    return m


# ─── Matplotlib Charts ──────────────────────────────────────────────────────────


def eda(
    targets_features: pd.DataFrame,
    target: str = "nearest_fire_km",
    use_cache: bool = True,
):
    save_path = FIGURES_DIR / f"eda_summary_{target}.png"

    if use_cache and save_path.exists():
        print(f"Loading cached EDA figure from {save_path}")
        img = plt.imread(str(save_path))
        fig, ax = plt.subplots(figsize=(16, 4), dpi=150)
        ax.imshow(img)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        return fig

    # Work with plain DataFrame — GeoDataFrame column ops are slow on large data
    df = pd.DataFrame(targets_features.drop(columns="geometry", errors="ignore"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes = axes.flatten()
    # 1. Target distribution
    axes[0].hist(df[target], bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Distance to Nearest Fire (km)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Target Distribution")

    # 2. Missing data summary by feature group
    census_cols_all = [c for c in df.columns if c.startswith("B")]
    missing_pct = df[census_cols_all].isna().mean() * 100
    axes[1].hist(
        missing_pct[missing_pct > 0],
        bins=20,
        edgecolor="black",
        alpha=0.7,
        color="orange",
    )
    axes[1].set_xlabel("% Missing")
    axes[1].set_ylabel("Number of Features")
    axes[1].set_title(f"Missingness Across {len(census_cols_all)} Census Features")

    # 3. Top 10 census feature correlations with target
    # Only use census columns (B*) — exclude proximity features derived from
    # the same wildfire data as the target to avoid circular correlations.
    census_numeric = [
        c for c in census_cols_all if c in df.select_dtypes(include="number").columns
    ]
    census_arr = df[census_numeric].values
    target_arr = df[target].values
    census_centered = census_arr - np.nanmean(census_arr, axis=0)
    target_centered = target_arr - np.nanmean(target_arr)
    cov = np.nanmean(census_centered * target_centered[:, None], axis=0)
    corr_values = cov / (np.nanstd(census_arr, axis=0) * np.nanstd(target_arr))
    corrs = pd.Series(corr_values, index=census_numeric)
    top_corrs = corrs.abs().nlargest(10)
    top_corrs.index = [
        census_code_to_label(c) if c.startswith("B") else c for c in top_corrs.index
    ]
    top_corrs.sort_values().plot.barh(ax=axes[2], color="steelblue")
    axes[2].set_xlabel("|Correlation| with Target")
    axes[2].set_title("Top 10 Correlated Features")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nDataset: {len(df)} properties, {df.shape[1]} columns")
    print(f"Census features: {len(census_cols_all)}")
    print(f"Features with missing data: {(missing_pct > 0).sum()}")
    print(f"Target range: {df[target].min():.1f} – {df[target].max():.1f} km")
    return fig


def plot_feature_importance(
    importances: pd.Series,
    title: str = "Top Feature Importances",
    figsize: tuple[int, int] = (10, 6),
    color: str = "steelblue",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a horizontal bar chart of feature importances.

    Parameters
    ----------
    importances : pd.Series
        Feature importances with feature names as index.
    title : str
        Chart title.
    figsize : tuple
        Figure size (width, height).
    color : str
        Bar color.
    save_path : Path, optional
        If provided, save the figure to this file.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by importance (ascending for horizontal bar chart)
    sorted_imp = importances.sort_values(ascending=True)

    # Use integer positions so duplicate labels each get their own bar
    y_pos = range(len(sorted_imp))
    ax.barh(y_pos, sorted_imp.values, color=color)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sorted_imp.index)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_actual_vs_predicted(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    title: str = "Actual vs Predicted",
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    figsize: tuple[int, int] = (8, 8),
    alpha: float = 0.6,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a scatter plot of actual vs predicted values with identity line.

    Parameters
    ----------
    y_actual : array-like
        Actual target values.
    y_predicted : array-like
        Predicted target values.
    title : str
        Chart title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : tuple
        Figure size (width, height).
    alpha : float
        Point transparency.
    save_path : Path, optional
        If provided, save the figure to this file.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_actual, y_predicted, alpha=alpha, edgecolors="none", s=50)

    # Identity line
    lims = [
        min(min(y_actual), min(y_predicted)),
        max(max(y_actual), max(y_predicted)),
    ]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add R² annotation
    from sklearn.metrics import r2_score

    r2 = r2_score(y_actual, y_predicted)
    ax.annotate(
        f"R² = {r2:.4f}",
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        ha="right",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_residuals(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    title: str = "Residuals Plot",
    figsize: tuple[int, int] = (10, 6),
    alpha: float = 0.6,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a residuals plot (predicted vs residuals).

    Parameters
    ----------
    y_actual : array-like
        Actual target values.
    y_predicted : array-like
        Predicted target values.
    title : str
        Chart title.
    figsize : tuple
        Figure size (width, height).
    alpha : float
        Point transparency.
    save_path : Path, optional
        If provided, save the figure to this file.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    residuals = y_actual - y_predicted

    ax.scatter(y_predicted, residuals, alpha=alpha, edgecolors="none", s=50)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig
