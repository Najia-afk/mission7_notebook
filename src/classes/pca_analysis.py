import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional

class PCAAnalysis:
    def __init__(self, df: pd.DataFrame, features: List[str] = None, n_components: int = 10):
        """
        Initialize PCA analysis with data
        
        Parameters:
        -----------
        df : DataFrame
            Feature data for analysis
        features : list, optional
            Features to use for PCA (defaults to all numeric columns)
        n_components : int, optional
            Maximum number of components to extract (default: 10)
        """
        self.df = df
        
        # Auto-detect numeric features if not specified
        if features is None:
            self.features = df.select_dtypes(include=['number']).columns.tolist()
        else:
            self.features = features
            
        # Apply PCA with explained variance ratio
        self.n_components = min(n_components, len(self.features))
        self.pca = PCA(n_components=self.n_components)
        self.X = self.pca.fit_transform(df[self.features])
        
        # Define color palette for consistency
        self.colors = [
            '#5cb85c',  # green
            '#5bc0de',  # blue
            '#f0ad4e',  # orange
            '#d9534f',  # red
            '#9370DB',  # purple
        ]
    
    def plot_explained_variance(self, max_components: int = 20) -> go.Figure:
        """
        Plot PCA explained variance by number of components.
        
        Parameters:
        -----------
        max_components : int
            Maximum number of components to analyze
        
        Returns:
        --------
        fig : plotly Figure
            Interactive plot of explained variance
        """
        # Limit max_components to number of features
        max_components = min(max_components, len(self.features))
        
        # Fit PCA with max components
        pca = PCA(n_components=max_components)
        pca.fit(self.df[self.features])
        
        # Get explained variance data
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        components = list(range(1, max_components + 1))
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add individual explained variance
        fig.add_trace(
            go.Bar(
                x=components, 
                y=explained_variance,
                name="Individual Explained Variance",
                marker_color='rgb(55, 83, 109)'
            ),
            secondary_y=False
        )
        
        # Add cumulative explained variance
        fig.add_trace(
            go.Scatter(
                x=components, 
                y=cumulative_variance,
                name="Cumulative Explained Variance",
                mode='lines+markers',
                line=dict(color='red', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title='PCA Explained Variance by Number of Components',
            xaxis_title='Number of Components',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
            width=900,
            height=600
        )
        
        fig.update_yaxes(title_text="Individual Explained Variance", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
        
        return fig
    
    def plot_biplot(self, n_features: int = 20, include_points: bool = False, sample_size: int = 1000) -> go.Figure:
        """
        Create a PCA biplot showing feature loadings on principal components.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to display by loading magnitude
        include_points : bool
            Whether to include data points (default: False)
        sample_size : int
            Number of observations to sample if include_points=True
            
        Returns:
        --------
        fig : plotly Figure
            Interactive biplot of feature loadings
        """
        # Get PCA loadings (feature weights for each component)
        loadings = self.pca.components_.T
        
        # Only take first 2 components for the biplot
        pc1_loadings = loadings[:, 0]
        pc2_loadings = loadings[:, 1]
        
        # Get feature names
        feature_names = self.features
        
        # Create a DataFrame with loadings
        loadings_df = pd.DataFrame(
            loadings[:, :2],
            columns=['PC1', 'PC2'],
            index=feature_names
        )
        
        # Sort features by magnitude of loading
        loadings_df['magnitude'] = np.sqrt(loadings_df['PC1']**2 + loadings_df['PC2']**2)
        loadings_df = loadings_df.sort_values('magnitude', ascending=False)
        
        # Select top features
        top_features = loadings_df.iloc[:n_features].index.tolist()
        
        # Set up figure
        fig = go.Figure()
        
        # Add scatter plot for feature loadings
        fig.add_trace(
            go.Scatter(
                x=loadings_df.loc[top_features, 'PC1'],
                y=loadings_df.loc[top_features, 'PC2'],
                mode='markers+text',
                text=top_features,
                textposition='top center',
                marker=dict(
                    size=10, 
                    color='rgba(55, 83, 109, 0.7)',
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name='Feature Loadings'
            )
        )
        
        # Add lines from origin to each point
        for feature in top_features:
            x = loadings_df.loc[feature, 'PC1']
            y = loadings_df.loc[feature, 'PC2']
            fig.add_shape(
                type='line',
                x0=0, y0=0, x1=x, y1=y,
                line=dict(color='rgba(55, 83, 109, 0.3)', width=1)
            )
        
        # Add a circle to represent correlation of 1
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='rgba(0,0,0,0.3)', width=1),
                showlegend=False
            )
        )
        
        # Add sample points if requested
        if include_points:
            # Sample points if too many
            if len(self.X) > sample_size:
                indices = np.random.choice(len(self.X), sample_size, replace=False)
                points = self.X[indices, :2]  # Only first 2 components
            else:
                points = self.X[:, :2]
                
            fig.add_trace(
                go.Scatter(
                    x=points[:, 0],
                    y=points[:, 1],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='rgba(70, 130, 180, 0.4)',
                        line=dict(width=0)
                    ),
                    name='Data Points',
                    hoverinfo='skip'
                )
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'PCA Biplot: Feature Loadings on Principal Components',
                'font': {'size': 18, 'family': "Arial, sans-serif"}
            },
            xaxis=dict(
                title=f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)',
                range=[-1.1, 1.1],
                zeroline=True, 
                zerolinewidth=1, 
                zerolinecolor='black',
                gridcolor='#EEEEEE'
            ),
            yaxis=dict(
                title=f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)',
                range=[-1.1, 1.1],
                zeroline=True, 
                zerolinewidth=1, 
                zerolinecolor='black',
                gridcolor='#EEEEEE'
            ),
            width=900,
            height=700,
            legend=dict(
                x=0.01, 
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            template='plotly_white'
        )
        
        return fig
        
    def plot_feature_importance(self, n_components: int = 3, top_n: int = 10) -> go.Figure:
        """
        Visualize the importance of each feature in the principal components.
        
        Parameters:
        -----------
        n_components : int
            Number of components to visualize
        top_n : int
            Number of top features to display (sorted by PC1 importance)
            
        Returns:
        --------
        fig : plotly Figure
            Feature importance heatmap
        """
        # Get feature names and loadings
        feature_names = self.features
        n_components = min(n_components, len(self.pca.components_))
        
        # Create a dataframe of loadings
        loadings_df = pd.DataFrame(
            data=self.pca.components_[:n_components, :].T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=feature_names
        )
        
        # Take absolute values for importance
        abs_loadings = np.abs(loadings_df)
        
        # Sort by PC1 absolute loading (or sum if PC1 not available for some reason)
        if 'PC1' in abs_loadings.columns:
            sorted_idx = abs_loadings['PC1'].sort_values(ascending=False).index
        else:
            sorted_idx = abs_loadings.sum(axis=1).sort_values(ascending=False).index
            
        # Filter top N
        sorted_idx = sorted_idx[:top_n]
        abs_loadings = abs_loadings.loc[sorted_idx]
        
        # Create heatmap
        fig = px.imshow(
            abs_loadings,
            labels=dict(x="Principal Component", y="Feature", color="Absolute Loading"),
            x=[f'PC{i+1}<br>({self.pca.explained_variance_ratio_[i]:.2%})' for i in range(n_components)],
            y=abs_loadings.index,
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance in Principal Components (Sorted by PC1)',
            width=900,
            height=max(500, 25 * len(abs_loadings)),
            coloraxis_colorbar=dict(title="Absolute Loading")
        )
        
        return fig
    
    def get_components(self, n_components: int = None) -> np.ndarray:
        """
        Get the PCA components for use in clustering algorithms
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to return (default: all available)
            
        Returns:
        --------
        X_components : numpy array
            PCA component values
        """
        if n_components is None:
            return self.X
        else:
            return self.X[:, :min(n_components, self.n_components)]
