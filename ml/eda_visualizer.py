"""
MLForge ML Engine - Matplotlib Visual Rendering Engine Module
Generates headless, base64-encoded PNG visualizations for histograms, box plots,
correlation heatmaps, missing value bar plots, target class distributions,
bivariate scatter plots, feature importances, and residual error distributions.
"""

import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple


class EDAVisualizer:
    """
    Headless Matplotlib chart rendering engine returning base64 PNG data URIs.
    """

    @staticmethod
    def render_histogram_grid(df: pd.DataFrame, max_cols: int = 6) -> Optional[str]:
        """
        Renders subplots of distribution histograms for numerical features.
        """
        num_cols = list(df.select_dtypes(include=[np.number]).columns[:max_cols])
        if not num_cols:
            return None

        n = len(num_cols)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, col in enumerate(num_cols):
            ax = axes[i]
            series = df[col].dropna()
            ax.hist(series, bins=20, color='#3b82f6', edgecolor='#1e3a8a', alpha=0.7)
            ax.set_title(col, fontsize=10, fontweight='bold', color='#f8fafc')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_facecolor('#0f172a')
            ax.tick_params(colors='#94a3b8')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        return EDAVisualizer._fig_to_base64(fig)

    @staticmethod
    def render_boxplot_grid(df: pd.DataFrame, max_cols: int = 6) -> Optional[str]:
        """
        Renders boxplots for detecting outliers in numerical features.
        """
        num_cols = list(df.select_dtypes(include=[np.number]).columns[:max_cols])
        if not num_cols:
            return None

        n = len(num_cols)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, col in enumerate(num_cols):
            ax = axes[i]
            series = df[col].dropna()
            ax.boxplot(series, patch_artist=True, boxprops=dict(facecolor='#8b5cf6', color='#6d28d9'), medianprops=dict(color='#f59e0b'))
            ax.set_title(f"Boxplot: {col}", fontsize=10, fontweight='bold', color='#f8fafc')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_facecolor('#0f172a')
            ax.tick_params(colors='#94a3b8')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        return EDAVisualizer._fig_to_base64(fig)

    @staticmethod
    def render_correlation_heatmap(df: pd.DataFrame, max_cols: int = 12) -> Optional[str]:
        """
        Renders Pearson correlation matrix heatmap.
        """
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty or len(num_df.columns) < 2:
            return None

        cols = list(num_df.columns[:max_cols])
        corr = num_df[cols].corr().fillna(0)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)

        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='left', color='#94a3b8', fontsize=8)
        ax.set_yticklabels(cols, color='#94a3b8', fontsize=8)

        for i in range(len(cols)):
            for j in range(len(cols)):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if abs(val) > 0.5 else 'black', fontsize=7)

        ax.set_title("Pearson Correlation Heatmap", fontsize=12, fontweight='bold', color='#f8fafc', pad=20)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        plt.tight_layout()
        return EDAVisualizer._fig_to_base64(fig)

    @staticmethod
    def render_missing_values_bar(df: pd.DataFrame) -> Optional[str]:
        """
        Renders bar chart of missing value counts per column.
        """
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(missing.index, missing.values, color='#ef4444', edgecolor='#b91c1c', alpha=0.8)
        ax.set_title("Missing Values Count by Column", fontsize=11, fontweight='bold', color='#f8fafc')
        ax.set_ylabel("Count of NaN Cells", color='#94a3b8')
        ax.tick_params(axis='x', rotation=45, colors='#94a3b8')
        ax.tick_params(axis='y', colors='#94a3b8')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_facecolor('#0f172a')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='#f8fafc', fontsize=8)

        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        return EDAVisualizer._fig_to_base64(fig)

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{data}"
