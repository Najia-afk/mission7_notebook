import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

class OutlierAnalyzer:
    """
    Class to analyze and visualize outliers in the dataset using multiple methods.
    Optimized for memory usage.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.methods = {
            "Z-score (±3)": {
                "function": lambda col: (
                    col.mean() - 3 * col.std(),
                    col.mean() + 3 * col.std()
                ),
                "description": "Mean ± 3×Std. Dev. (common statistical approach)",
                "color": "rgba(148, 103, 189, 0.7)"
            },
            "IQR 1.5": {
                "function": lambda col: (
                    col.quantile(0.25) - 1.5 * (col.quantile(0.75) - col.quantile(0.25)),
                    col.quantile(0.75) + 1.5 * (col.quantile(0.75) - col.quantile(0.25))
                ),
                "description": "Standard method: Q1-1.5×IQR to Q3+1.5×IQR",
                "color": "rgba(31, 119, 180, 0.7)"
            },
            "Z-score (±2)": {
                "function": lambda col: (
                    col.mean() - 2 * col.std(),
                    col.mean() + 2 * col.std()
                ),
                "description": "Mean ± 2×Std. Dev. (stricter approach)",
                "color": "rgba(140, 86, 75, 0.7)"
            },
            "Quantile 1-99": {
                "function": lambda col: (col.quantile(0.01), col.quantile(0.99)),
                "description": "1st to 99th percentile range",
                "color": "rgba(255, 127, 14, 0.7)"
            },
            "Quantile 5-95": {
                "function": lambda col: (col.quantile(0.05), col.quantile(0.95)),
                "description": "5th to 95th percentile range",
                "color": "rgba(44, 160, 44, 0.7)"
            },
            "Full Range": {
                "function": lambda col: (col.min(), col.max()),
                "description": "No outlier filtering (min to max)",
                "color": "rgba(214, 39, 40, 0.7)"
            }
        }

    def analyze_outliers(self, columns: list = None):
        """
        Analyze outliers in numerical data using different detection methods.
        Returns summaries and info, but NOT full cleaned dataframes to save memory.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        all_summaries = {}
        all_outlier_info = {}
        all_stats_info = {}
        
        for method_name, method_info in self.methods.items():
            outlier_info = {}
            stats_info = {}
            
            for col in columns:
                if col not in self.df.columns or self.df[col].count() == 0:
                    continue
                    
                lower_bound, upper_bound = method_info["function"](self.df[col])
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
                
                mean = self.df[col].mean()
                # Calculate clean mean without creating a full copy
                clean_series = self.df[col][~outliers]
                clean_mean = clean_series.mean() if not clean_series.empty else np.nan
                
                if pd.notna(clean_mean) and clean_mean != 0:  
                    mean_percent_change = ((mean - clean_mean) / clean_mean) * 100
                else:
                    mean_percent_change = 0
                
                stats_info[col] = {
                    'mean': mean,
                    'clean_mean': clean_mean,
                    'mean_percent_change': mean_percent_change,
                    'skewness': self.df[col].skew()
                }
                
                outlier_info[col] = {
                    'outlier_count': outliers.sum(),
                    'outlier_percentage': outliers.sum() / self.df[col].count() * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'num_below_lower': (outliers & (self.df[col] < lower_bound)).sum(),
                    'num_above_upper': (outliers & (self.df[col] > upper_bound)).sum()
                }
            
            summary_df = pd.DataFrame({
                'Variable': [col for col in outlier_info.keys()],
                'Outlier Count': [info['outlier_count'] for info in outlier_info.values()],
                'Outlier %': [f"{info['outlier_percentage']:.2f}%" for info in outlier_info.values()],
                'Min Limit': [info['lower_bound'] for info in outlier_info.values()],
                'Max Limit': [info['upper_bound'] for info in outlier_info.values()],
                'Mean % Change': [stats_info[col]['mean_percent_change'] for col in outlier_info.keys()],
                'Skewness': [stats_info[col]['skewness'] for col in outlier_info.keys()]
            })
            
            if not summary_df.empty:
                summary_df = summary_df.sort_values(by='Outlier Count', ascending=False)
            
            all_summaries[method_name] = summary_df
            all_outlier_info[method_name] = outlier_info
            all_stats_info[method_name] = stats_info
            
        return all_summaries, all_outlier_info, all_stats_info

    def get_cleaned_dataframe(self, method_name: str, columns: list = None) -> pd.DataFrame:
        """
        Returns a cleaned dataframe for a specific method.
        """
        if method_name not in self.methods:
            raise ValueError(f"Method {method_name} not found.")
            
        if columns is None:
            columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
        df_clean = self.df.copy()
        method_info = self.methods[method_name]
        
        for col in columns:
            if col not in self.df.columns:
                continue
            lower_bound, upper_bound = method_info["function"](self.df[col])
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
            df_clean.loc[outliers, col] = np.nan
            
        return df_clean

    def plot_outlier_summary(self, all_summaries):
        """Create an interactive outlier visualization with method dropdown"""
        if not all_summaries:
            print("No summaries available to plot.")
            return None

        method_names = list(all_summaries.keys())
        default_method = method_names[0]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Outlier Counts by Variable", "Impact on Mean Values (% Change)"),
            vertical_spacing=0.3,
            specs=[[{"type": "bar"}], [{"type": "bar"}]]
        )
        
        for method_name in method_names:
            summary_df = all_summaries[method_name]
            visible = (method_name == default_method)
            
            fig.add_trace(
                go.Bar(
                    x=summary_df['Variable'],
                    y=summary_df['Outlier Count'],
                    name='Outlier Count',
                    marker_color='red',
                    text=summary_df['Outlier %'],
                    hovertemplate='%{x}<br>Outliers: %{y}<br>(%{text})<extra></extra>',
                    visible=visible
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=summary_df['Variable'],
                    y=summary_df['Mean % Change'],
                    name='Mean % Change',
                    marker_color='purple',
                    text=[f"{x:.1f}%" for x in summary_df['Mean % Change']],
                    textposition='auto',
                    hovertemplate='%{x}<br>Mean % Change: %{text}<extra></extra>',
                    visible=visible
                ),
                row=2, col=1
            )
        
        dropdown_buttons = []
        for i, method_name in enumerate(method_names):
            visibility = [method == method_name for method in method_names for _ in range(2)]
            
            dropdown_buttons.append(
                dict(
                    label=f"{method_name} - {self.methods[method_name]['description']}",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"Outlier Analysis using {method_name} Method"}
                    ]
                )
            )
        
        fig.update_layout(
            title={
                'text': f"Outlier Analysis using {default_method} Method",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=800,
            showlegend=False,
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.7,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            margin=dict(r=200),
            template='plotly_white'
        )
        
        fig.update_xaxes(tickangle=45)
        return fig

    def compare_variable_outliers(self, variable_name: str):
        """
        Create detailed visualization comparing outlier detection methods for a specific variable.
        """
        if variable_name not in self.df.columns:
            print(f"Variable '{variable_name}' not found in dataframe.")
            return None
        
        data = self.df[variable_name].dropna()
        method_stats = {}
        
        for method_name, method_info in self.methods.items():
            lower_bound, upper_bound = method_info["function"](data)
            outliers = ((data < lower_bound) | (data > upper_bound))
            clean_data = data[~outliers]
            
            method_stats[method_name] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outliers.sum(),
                "outlier_percentage": (outliers.sum() / len(data)) * 100,
                "mean_with_outliers": data.mean(),
                "mean_without_outliers": clean_data.mean() if len(clean_data) > 0 else np.nan,
                "std_with_outliers": data.std(),
                "std_without_outliers": clean_data.std() if len(clean_data) > 0 else np.nan,
                "color": method_info["color"]
            }
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Distribution of {variable_name} with Different Bounds",
                "Outlier Counts by Method",
                "Impact on Mean",
                "Impact on Standard Deviation"
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )
        
        # 1. Distribution
        fig.add_trace(
            go.Histogram(x=data, nbinsx=30, name="Distribution", marker_color="rgba(100, 100, 100, 0.5)", opacity=0.7),
            row=1, col=1
        )
        
        for method_name, stats in method_stats.items():
            fig.add_trace(
                go.Scatter(x=[stats["lower_bound"], stats["lower_bound"]], y=[0, data.value_counts().max()], mode="lines", name=f"{method_name} Lower", line=dict(color=stats["color"], width=2, dash="dash")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=[stats["upper_bound"], stats["upper_bound"]], y=[0, data.value_counts().max()], mode="lines", name=f"{method_name} Upper", line=dict(color=stats["color"], width=2, dash="dash")),
                row=1, col=1
            )
            
        # 2. Counts
        fig.add_trace(
            go.Bar(x=list(self.methods.keys()), y=[stats["outlier_count"] for stats in method_stats.values()], marker_color=[stats["color"] for stats in method_stats.values()], text=[f"{stats['outlier_percentage']:.1f}%" for stats in method_stats.values()], textposition="auto", name="Outlier Count"),
            row=1, col=2
        )
        
        # 3. Means
        fig.add_trace(go.Bar(x=list(self.methods.keys()), y=[stats["mean_with_outliers"] for stats in method_stats.values()], name="With Outliers", marker_color="rgba(31, 119, 180, 0.7)"), row=2, col=1)
        fig.add_trace(go.Bar(x=list(self.methods.keys()), y=[stats["mean_without_outliers"] for stats in method_stats.values()], name="Without Outliers", marker_color="rgba(255, 127, 14, 0.7)"), row=2, col=1)
        
        # 4. Stds
        fig.add_trace(go.Bar(x=list(self.methods.keys()), y=[stats["std_with_outliers"] for stats in method_stats.values()], name="With Outliers", marker_color="rgba(31, 119, 180, 0.7)"), row=2, col=2)
        fig.add_trace(go.Bar(x=list(self.methods.keys()), y=[stats["std_without_outliers"] for stats in method_stats.values()], name="Without Outliers", marker_color="rgba(255, 127, 14, 0.7)"), row=2, col=2)
        
        fig.update_layout(height=800, width=1200, title_text=f"Detailed Outlier Analysis for {variable_name}", showlegend=False, template='plotly_white')
        return fig
