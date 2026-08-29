"""
MLForge ML Engine - Exploratory Data Analysis (EDA) Module
Calculates dataset statistical summaries, quantile metrics, correlation matrices,
and generates local Matplotlib chart visualizations (histograms, box plots, correlation heatmaps, missing value bar charts).
"""

import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils.helpers import make_json_serializable


class ExploratoryDataAnalysis:
    """
    EDA analysis engine and chart visualization generator.
    """

    @staticmethod
    def get_numerical_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates numerical statistical metrics: mean, std, min, quantiles (25%, 50%, 75%), max, skewness.
        """
        num_df = df.select_dtypes(include=[np.number])
        stats = {}
        
        for col in num_df.columns:
            series = num_df[col].dropna()
            if series.empty:
                continue
                
            stats[col] = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "min": round(float(series.min()), 4),
                "q25": round(float(series.quantile(0.25)), 4),
                "median": round(float(series.median()), 4),
                "q75": round(float(series.quantile(0.75)), 4),
                "max": round(float(series.max()), 4),
                "skewness": round(float(series.skew()), 4) if len(series) > 2 else 0.0
            }
            
        return stats

    @staticmethod
    def get_categorical_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates categorical frequency statistics: unique count, top value, top frequency, value counts.
        """
        cat_df = df.select_dtypes(include=['object', 'category', 'bool'])
        stats = {}
        
        for col in cat_df.columns:
            series = cat_df[col].dropna()
            if series.empty:
                continue
                
            value_counts = series.value_counts().head(10).to_dict()
            top_val = str(series.mode().iloc[0]) if not series.empty else ""
            top_freq = int(series.value_counts().iloc[0]) if not series.empty else 0
            
            stats[col] = {
                "count": int(len(series)),
                "unique": int(series.nunique()),
                "top": top_val,
                "freq": top_freq,
                "top_value_counts": make_json_serializable(value_counts)
            }
            
        return stats

    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates Pearson correlation matrix for numerical features.
        """
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty or len(num_df.columns) < 2:
            return {"columns": [], "matrix": []}
            
        corr = num_df.corr().round(4).fillna(0)
        return {
            "columns": list(corr.columns),
            "matrix": corr.values.tolist()
        }

    @staticmethod
    def generate_chart_base64(fig) -> str:
        """
        Converts Matplotlib figure into a base64 encoded PNG image string for inline web rendering.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def create_distribution_chart(df: pd.DataFrame, max_cols: int = 4) -> Optional[str]:
        """
        Generates grid histogram plot for numerical features.
        """
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return None
            
        cols = list(num_df.columns)[:max_cols]
        fig, axes = plt.subplots(len(cols), 1, figsize=(7, 2.5 * len(cols)), squeeze=False)
        fig.patch.set_facecolor('#161e2e')
        
        for i, col in enumerate(cols):
            ax = axes[i, 0]
            ax.set_facecolor('#111827')
            ax.hist(num_df[col].dropna(), bins=20, color='#3b82f6', edgecolor='#1e293b', alpha=0.85)
            ax.set_title(f"Distribution: {col}", color='#f9fafb', fontsize=10, fontweight='bold')
            ax.tick_params(colors='#9ca3af', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#1e293b')
            ax.spines['bottom'].set_color('#1e293b')
            ax.grid(True, linestyle='--', alpha=0.2)
            
        plt.tight_layout()
        return ExploratoryDataAnalysis.generate_chart_base64(fig)

    @staticmethod
    def create_box_plot_chart(df: pd.DataFrame, max_cols: int = 4) -> Optional[str]:
        """
        Generates box plot chart for numerical feature outlier inspection.
        """
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return None
            
        cols = list(num_df.columns)[:max_cols]
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#161e2e')
        ax.set_facecolor('#111827')
        
        # Standardize for comparison boxplot
        data_to_plot = []
        for col in cols:
            series = num_df[col].dropna()
            std = series.std()
            normalized = (series - series.mean()) / (std if std != 0 else 1.0)
            data_to_plot.append(normalized)
            
        bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=cols)
        for patch in bp['boxes']:
            patch.set_facecolor('#8b5cf6')
            patch.set_alpha(0.7)
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='#f9fafb')
            
        ax.set_title("Standardized Numerical Outlier Box Plot (Z-Score)", color='#f9fafb', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#9ca3af', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.2)
        
        plt.tight_layout()
        return ExploratoryDataAnalysis.generate_chart_base64(fig)

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame) -> Optional[str]:
        """
        Generates correlation matrix heatmap image.
        """
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty or len(num_df.columns) < 2:
            return None
            
        cols = list(num_df.columns)[:8]
        corr = num_df[cols].corr().fillna(0).values
        
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor('#161e2e')
        ax.set_facecolor('#111827')
        
        im = ax.imshow(corr, cmap='Blues', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, color='#9ca3af', rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(cols, color='#9ca3af', fontsize=9)
        
        # Annotate matrix values
        for i in range(len(cols)):
            for j in range(len(cols)):
                text = ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="#f9fafb" if abs(corr[i, j]) > 0.5 else "#9ca3af", fontsize=8)
                
        ax.set_title("Pearson Correlation Heatmap Matrix", color='#f9fafb', fontsize=11, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='#9ca3af')
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#9ca3af')
        
        plt.tight_layout()
        return ExploratoryDataAnalysis.generate_chart_base64(fig)

    @staticmethod
    def create_missing_values_chart(df: pd.DataFrame) -> Optional[str]:
        """
        Generates missing values bar chart per feature column.
        """
        null_counts = df.isna().sum()
        null_counts = null_counts[null_counts > 0]
        
        if null_counts.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#161e2e')
        ax.set_facecolor('#111827')
        
        ax.bar(null_counts.index, null_counts.values, color='#f59e0b', alpha=0.85)
        ax.set_title("Missing Values Count Per Feature Column", color='#f9fafb', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#9ca3af', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.2)
        
        plt.tight_layout()
        return ExploratoryDataAnalysis.generate_chart_base64(fig)

    @staticmethod
    def create_target_distribution_chart(df: pd.DataFrame, target_col: str) -> Optional[str]:
        """
        Generates class / target distribution chart.
        """
        if not target_col or target_col not in df.columns:
            return None
            
        series = df[target_col].dropna()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#161e2e')
        ax.set_facecolor('#111827')
        
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 15:
            # Continuous numerical target histogram
            ax.hist(series, bins=20, color='#10b981', alpha=0.85)
            ax.set_title(f"Target Distribution (Continuous Regression): '{target_col}'", color='#f9fafb', fontsize=10, fontweight='bold')
        else:
            # Categorical / discrete target bar plot
            counts = series.value_counts().head(10)
            ax.bar([str(k) for k in counts.index], counts.values, color='#10b981', alpha=0.85)
            ax.set_title(f"Target Class Distribution (Classification): '{target_col}'", color='#f9fafb', fontsize=10, fontweight='bold')
            
        ax.tick_params(colors='#9ca3af', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.2)
        
        plt.tight_layout()
        return ExploratoryDataAnalysis.generate_chart_base64(fig)
