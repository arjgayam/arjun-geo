"""
Visualization helpers — choropleth maps and scatter plots.
All functions save a PNG and print the output path.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_choropleth(gdf, column, title, cmap, output_path, legend_label=""):
    """Draw a choropleth map of *column* and save to *output_path*."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_data = gdf.dropna(subset=[column])
    plot_data.plot(
        column=column,
        cmap=cmap,
        linewidth=0.3,
        edgecolor="gray",
        legend=True,
        legend_kwds={"label": legend_label, "shrink": 0.6},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scatter_with_trend(df, x_col, y_col, title, output_path,
                            xlabel="", ylabel=""):
    """Scatter plot with a linear trend line and slope annotation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    clean = df[[x_col, y_col]].dropna()

    ax.scatter(clean[x_col], clean[y_col],
               alpha=0.6, edgecolors="k", linewidth=0.3, s=30, color="#2196F3")

    # Trend line
    z = np.polyfit(clean[x_col], clean[y_col], 1)
    xs = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), "r--", linewidth=1.5,
            label=f"Trend (slope = {z[0]:.2f})")

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_observed_vs_predicted(df, output_path):
    """Scatter of observed vs. predicted temperature with a 1:1 reference line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["mean_temp"], df["predicted_temp"],
               alpha=0.6, edgecolors="k", linewidth=0.3, s=30, color="#FF9800")

    vals = np.concatenate([df["mean_temp"].values, df["predicted_temp"].values])
    lo, hi = vals.min() - 0.5, vals.max() + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="1:1 line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Observed Mean Temperature (°C)")
    ax.set_ylabel("Predicted Mean Temperature (°C)")
    ax.set_title("Observed vs. Predicted Temperature by Census Tract")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
