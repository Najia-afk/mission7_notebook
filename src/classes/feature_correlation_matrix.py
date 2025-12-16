import numpy as np
import plotly.graph_objects as go
import pandas as pd

class CorrelationAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.corr_matrix = None
    
    def compute_correlation(self) -> pd.DataFrame:
        """Compute correlation matrix."""
        self.corr_matrix = self.df[self.numerical_features].corr()
        return self.corr_matrix
    
    def get_lower_triangle(self) -> pd.DataFrame:
        """Return lower triangle of correlation matrix."""
        if self.corr_matrix is None:
            self.compute_correlation()
        # Create mask for lower triangle (excluding diagonal)
        mask = np.tril(np.ones_like(self.corr_matrix), k=-1)
        # Apply mask to correlation matrix
        return self.corr_matrix * mask
    
    def plot_correlation_matrix(self) -> go.Figure:
        """Generate correlation matrix heatmap showing only lower triangle."""
        if self.corr_matrix is None:
            self.compute_correlation()
        
        # Get lower triangle
        lower_triangle = self.get_lower_triangle()
        
        # Create heatmap with lower triangle only
        fig = go.Figure(data=go.Heatmap(
            z=lower_triangle,
            x=self.numerical_features,
            y=self.numerical_features,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(lower_triangle, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            showscale=True
        ))
        
        fig.update_layout(
            title='Feature Correlation Analysis - Lower Triangle',
            xaxis={'tickangle': 45},
            yaxis={'tickangle': 0},
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        return fig