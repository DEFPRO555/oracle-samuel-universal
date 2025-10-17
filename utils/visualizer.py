import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import logging

class RealEstateVisualizer:
    """
    Comprehensive visualization utility for real estate data analysis.
    Supports both static matplotlib/seaborn and interactive plotly visualizations.
    """
    
    def __init__(self, df: pd.DataFrame = None):
        self.logger = logging.getLogger(__name__)
        self.df = df
        self.set_style()
    
    def set_style(self):
        """Set consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set plotly template
        self.plotly_template = "plotly_white"
    
    def plot_price_distribution(self, df: pd.DataFrame, price_column: str = 'price', 
                              figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create a comprehensive price distribution visualization.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(df[price_column], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title(f'Distribution of {price_column.title()}')
        axes[0].set_xlabel(price_column.title())
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df[price_column], patch_artist=True, 
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[1].set_title(f'{price_column.title()} Box Plot')
        axes[1].set_ylabel(price_column.title())
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_price_vs_features(self, df: pd.DataFrame, price_column: str = 'price',
                              feature_columns: List[str] = None, 
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create scatter plots of price vs various features.
        
        Args:
            df: DataFrame with data
            price_column: Name of price column
            feature_columns: List of feature columns to plot
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        if feature_columns is None:
            # Select numeric columns excluding price
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if price_column in feature_columns:
                feature_columns.remove(price_column)
        
        # Calculate subplot grid
        n_features = len(feature_columns)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(feature_columns):
            if i < len(axes):
                axes[i].scatter(df[feature], df[price_column], alpha=0.6, s=20)
                axes[i].set_xlabel(feature.title())
                axes[i].set_ylabel(price_column.title())
                axes[i].set_title(f'{price_column.title()} vs {feature.title()}')
                axes[i].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = df[feature].corr(df[price_column])
                axes[i].text(0.05, 0.95, f'r = {corr:.3f}', 
                           transform=axes[i].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(feature_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a correlation heatmap for numeric columns.
        
        Args:
            df: DataFrame with data
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        ax.set_title('Feature Correlation Heatmap')
        plt.tight_layout()
        return fig
    
    def plot_prediction_vs_actual(self, y_actual: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "Model", 
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create prediction vs actual scatter plot with performance metrics.
        
        Args:
            y_actual: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot
        axes[0].scatter(y_actual, y_pred, alpha=0.6, s=20)
        axes[0].plot([y_actual.min(), y_actual.max()], 
                    [y_actual.min(), y_actual.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Calculate metrics
        mse = np.mean((y_actual - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_actual - y_pred))
        r2 = 1 - (np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2))
        
        # Residuals plot
        residuals = y_actual - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{model_name}: Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
        axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_price_plot(self, df: pd.DataFrame, price_column: str = 'price',
                                    feature_column: str = 'area') -> go.Figure:
        """
        Create an interactive plotly scatter plot.

        Args:
            df: DataFrame with data
            price_column: Name of price column
            feature_column: Name of feature column for x-axis

        Returns:
            Plotly figure
        """
        fig = px.scatter(df, x=feature_column, y=price_column,
                        title=f'{price_column.title()} vs {feature_column.title()}',
                        labels={price_column: price_column.title(),
                               feature_column: feature_column.title()},
                        template=self.plotly_template)

        # Add trend line
        fig.add_traces(px.scatter(df, x=feature_column, y=price_column,
                                 trendline="ols").data[1])

        fig.update_layout(
            xaxis_title=feature_column.title(),
            yaxis_title=price_column.title(),
            hovermode='closest'
        )

        return fig

    def plot_scatter(self, x_col: str, y_col: str) -> go.Figure:
        """
        Create an interactive scatter plot between two columns.

        Args:
            x_col: Name of x-axis column
            y_col: Name of y-axis column

        Returns:
            Plotly figure
        """
        if self.df is None:
            # Return empty figure if no data
            return go.Figure()

        # Check for duplicate columns and handle them
        df_clean = self.df.copy()
        if df_clean.columns.duplicated().any():
            self.logger.warning("Duplicate columns detected, removing duplicates")
            # Keep only the first occurrence of each column
            df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

        fig = px.scatter(
            df_clean,
            x=x_col,
            y=y_col,
            title=f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}',
            labels={
                x_col: x_col.replace("_", " ").title(),
                y_col: y_col.replace("_", " ").title()
            },
            template=self.plotly_template,
            trendline="ols",
            trendline_color_override="red"
        )

        fig.update_layout(
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=y_col.replace("_", " ").title(),
            hovermode='closest',
            height=500
        )

        return fig

    def plot_box_plot(self, column: str) -> go.Figure:
        """
        Create an interactive box plot for a column.

        Args:
            column: Name of column to plot

        Returns:
            Plotly figure
        """
        if self.df is None:
            # Return empty figure if no data
            return go.Figure()

        # Check for duplicate columns and handle them
        df_clean = self.df.copy()
        if df_clean.columns.duplicated().any():
            self.logger.warning("Duplicate columns detected, removing duplicates")
            # Keep only the first occurrence of each column
            df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

        fig = go.Figure()

        fig.add_trace(go.Box(
            y=df_clean[column],
            name=column.replace("_", " ").title(),
            boxmean='sd',  # Show mean and standard deviation
            marker_color='lightblue',
            boxpoints='outliers'  # Show outliers
        ))

        fig.update_layout(
            title=f'Distribution of {column.replace("_", " ").title()}',
            yaxis_title=column.replace("_", " ").title(),
            template=self.plotly_template,
            height=500,
            showlegend=False
        )

        return fig
    
    def create_interactive_correlation_plot(self, df: pd.DataFrame) -> go.Figure:
        """
        Create an interactive correlation heatmap using plotly.
        
        Args:
            df: DataFrame with data
            
        Returns:
            Plotly figure
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Interactive Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features',
            template=self.plotly_template
        )
        
        return fig
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float],
                                     title: str = "Feature Importance",
                                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a horizontal bar plot of feature importance.

        Args:
            feature_importance: Dictionary with feature names and importance scores
            title: Plot title
            figsize: Figure size tuple

        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(features, importances, color='skyblue', alpha=0.7)

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center')

        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def plot_feature_importance(self, feature_importance: pd.DataFrame) -> go.Figure:
        """
        Create an interactive plotly bar chart of feature importance.

        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns

        Returns:
            Plotly figure
        """
        if feature_importance is None or feature_importance.empty:
            return go.Figure()

        # Sort by importance
        fi_sorted = feature_importance.sort_values('importance', ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=fi_sorted['importance'],
            y=fi_sorted['feature'],
            orientation='h',
            marker_color='lightblue',
            text=fi_sorted['importance'].round(3),
            textposition='outside'
        ))

        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template=self.plotly_template,
            height=max(400, len(fi_sorted) * 30),
            showlegend=False
        )

        return fig

    def plot_actual_vs_predicted(self, y_actual: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """
        Create an interactive scatter plot of actual vs predicted values.

        Args:
            y_actual: Actual values
            y_pred: Predicted values

        Returns:
            Plotly figure
        """
        # Calculate R² score
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        fig = go.Figure()

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_actual,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                color='lightblue',
                size=8,
                opacity=0.6
            ),
            text=[f'Actual: {a:.0f}<br>Predicted: {p:.0f}' for a, p in zip(y_actual, y_pred)],
            hovertemplate='%{text}<extra></extra>'
        ))

        # Perfect prediction line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))

        fig.update_layout(
            title=f'Actual vs Predicted Values (R² = {r2:.4f})',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            template=self.plotly_template,
            height=500,
            hovermode='closest'
        )

        return fig

    def plot_residuals(self, y_actual: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """
        Create an interactive residual plot.

        Args:
            y_actual: Actual values
            y_pred: Predicted values

        Returns:
            Plotly figure
        """
        residuals = y_actual - y_pred

        fig = go.Figure()

        # Residual scatter plot
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color='lightcoral',
                size=8,
                opacity=0.6
            ),
            text=[f'Predicted: {p:.0f}<br>Residual: {r:.0f}' for p, r in zip(y_pred, residuals)],
            hovertemplate='%{text}<extra></extra>'
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)

        fig.update_layout(
            title='Residual Plot',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            template=self.plotly_template,
            height=500,
            hovermode='closest'
        )

        return fig
    
    def create_model_comparison_plot(self, model_metrics: Dict[str, Dict[str, float]],
                                   metric: str = 'r2',
                                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a bar plot comparing model performance.
        
        Args:
            model_metrics: Dictionary with model names and their metrics
            metric: Metric to compare (r2, rmse, mae, etc.)
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        model_names = list(model_metrics.keys())
        metric_values = [model_metrics[name].get(metric, 0) for name in model_names]
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(model_names, metric_values, color='lightcoral', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Model Comparison - {metric.upper()}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if len(max(model_names, key=len)) > 10:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def create_dashboard_plots(self, df: pd.DataFrame, price_column: str = 'price') -> Dict[str, Any]:
        """
        Create a comprehensive set of plots for dashboard display.
        
        Args:
            df: DataFrame with data
            price_column: Name of price column
            
        Returns:
            Dictionary with plot objects
        """
        plots = {}
        
        # Price distribution
        plots['price_distribution'] = self.plot_price_distribution(df, price_column)
        
        # Correlation heatmap
        plots['correlation_heatmap'] = self.plot_correlation_heatmap(df)
        
        # Price vs features (top 3 numeric features)
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if price_column in numeric_features:
            numeric_features.remove(price_column)
        
        if len(numeric_features) >= 3:
            top_features = numeric_features[:3]
            plots['price_vs_features'] = self.plot_price_vs_features(df, price_column, top_features)
        
        # Interactive plots
        if len(numeric_features) > 0:
            plots['interactive_price_plot'] = self.create_interactive_price_plot(
                df, price_column, numeric_features[0])
            plots['interactive_correlation'] = self.create_interactive_correlation_plot(df)
        
        return plots
    
    def display_streamlit_plots(self, plots: Dict[str, Any]):
        """
        Display plots in Streamlit interface.
        
        Args:
            plots: Dictionary with plot objects
        """
        # Static plots
        if 'price_distribution' in plots:
            st.pyplot(plots['price_distribution'])
        
        if 'correlation_heatmap' in plots:
            st.pyplot(plots['correlation_heatmap'])
        
        if 'price_vs_features' in plots:
            st.pyplot(plots['price_vs_features'])
        
        # Interactive plots
        if 'interactive_price_plot' in plots:
            st.plotly_chart(plots['interactive_price_plot'], use_container_width=True)
        
        if 'interactive_correlation' in plots:
            st.plotly_chart(plots['interactive_correlation'], use_container_width=True)
