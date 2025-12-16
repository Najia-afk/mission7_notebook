import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import numpy as np
from kneed import KneeLocator
from tqdm.notebook import tqdm
from sklearn.inspection import permutation_importance

class DBSCANClusterAnalysis:
    def __init__(self, df: pd.DataFrame, features: List[str] = None, transformer = None, pca_components: np.ndarray = None):
        """
        Initialize DBSCAN clustering analysis
        
        Parameters:
        -----------
        df : DataFrame
            Feature data for clustering
        features : list, optional
            Features to use for clustering (defaults to all numeric columns)
        transformer : object, optional
            Transformer with inverse_transform method
        pca_components : numpy array, optional
            PCA components to use for visualization (if provided)
        """
        self.df = df
        self.transformer = transformer
        
        # Auto-detect numeric features if not specified
        if features is None:
            self.features = df.select_dtypes(include=['number']).columns.tolist()
        else:
            self.features = features
            
        # Store PCA components if provided
        self.pca_components = pca_components
        
        # Results storage
        self.clustering_results = {}
        
        # Create a consistent color palette
        self.cluster_colors = [
            '#5cb85c',  # green
            '#5bc0de',  # blue
            '#f0ad4e',  # orange
            '#d9534f',  # red
            '#9370DB',  # purple
            '#C71585',  # magenta
            '#20B2AA',  # teal
            '#F08080',  # coral
            '#4682B4',  # steel blue
            '#FFD700',  # gold
        ]
        # Special color for noise points
        self.noise_color = '#aaaaaa'  # Gray
        
        # Define transparent versions for fill areas
        self.cluster_colors_transparent = [
            f'rgba(92, 184, 92, 0.3)',    # green
            f'rgba(91, 192, 222, 0.3)',   # blue
            f'rgba(240, 173, 78, 0.3)',   # orange
            f'rgba(217, 83, 79, 0.3)',    # red
            f'rgba(147, 112, 219, 0.3)',  # purple
            f'rgba(199, 21, 133, 0.3)',   # magenta
            f'rgba(32, 178, 170, 0.3)',   # teal
            f'rgba(240, 128, 128, 0.3)',  # coral
            f'rgba(70, 130, 180, 0.3)',   # steel blue
            f'rgba(255, 215, 0, 0.3)',    # gold
        ]
    
    def get_cluster_name(self, cluster_idx: int) -> str:
        """Return a consistent name for a cluster index"""
        if cluster_idx == -1:
            return "Noise"
        return f'Cluster {cluster_idx}'
    
    def get_cluster_color(self, cluster_idx: int, transparent: bool = False) -> str:
        """Return a consistent color for a cluster index"""
        if cluster_idx == -1:
            return self.noise_color
            
        color_list = self.cluster_colors_transparent if transparent else self.cluster_colors
        return color_list[cluster_idx % len(color_list)]
    
    def find_optimal_eps(self, min_samples: int = 5, n_neighbors: int = 10, sample_size: int = 10000) -> go.Figure:
        """
        Find optimal epsilon value for DBSCAN using k-distance graph
        
        Parameters:
        -----------
        min_samples : int
            MinPts parameter for DBSCAN
        n_neighbors : int
            Number of neighbors for k-distance calculation
        sample_size : int
            Number of samples to use for faster computation
            
        Returns:
        --------
        fig : plotly Figure
            K-distance graph with suggested eps value
        """
        # Decide what data to use
        if self.pca_components is not None:
            X = self.pca_components[:, :3]  # Use first 3 PCA components
            print(f"Using PCA components for eps optimization")
        else:
            X = self.df[self.features].values
            print(f"Using original features for eps optimization")
        
        # Sample data if it's too large
        if len(X) > sample_size:
            print(f"Sampling {sample_size} records from {len(X)} for faster computation...")
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Calculate distances to k-nearest neighbors
        print("Calculating k-nearest neighbors distances...")
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_sample)
        distances, indices = nbrs.kneighbors(X_sample)
        
        # Sort distances to nth neighbor (last column)
        distance_desc = np.sort(distances[:, min_samples-1])[::-1]
        
        # Try to find the knee/elbow point
        try:
            i_knee = KneeLocator(
                range(len(distance_desc)), 
                distance_desc,
                curve='convex', 
                direction='decreasing'
            ).knee
            
            if i_knee is not None:
                optimal_eps = distance_desc[i_knee]
                print(f"Suggested optimal eps value: {optimal_eps:.4f}")
            else:
                optimal_eps = None
                print("Could not automatically determine optimal eps value")
        except Exception as e:
            print(f"Error in finding knee point: {e}")
            optimal_eps = None
        
        # Create k-distance plot
        fig = go.Figure()
        
        # Add distance curve
        fig.add_trace(
            go.Scatter(
                x=list(range(len(distance_desc))),
                y=distance_desc,
                mode='lines',
                line=dict(color=self.cluster_colors[0], width=2),
                name=f'Distance to {min_samples}th neighbor'
            )
        )
        
        # Add knee point if found
        if optimal_eps is not None:
            fig.add_trace(
                go.Scatter(
                    x=[i_knee],
                    y=[optimal_eps],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle-open',
                        line=dict(width=2, color='red')
                    ),
                    name=f'Optimal Eps: {optimal_eps:.4f}'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f'K-Distance Graph for DBSCAN Parameter Selection (min_samples={min_samples})',
            xaxis_title='Points sorted by distance',
            yaxis_title=f'Distance to {min_samples}th neighbor',
            width=900,
            height=600,
            showlegend=True
        )
        
        # Store the optimal eps value
        self.clustering_results['optimal_eps'] = optimal_eps if optimal_eps is not None else 0.5
        
        return fig
    
    def fit_dbscan(self, eps: float = None, min_samples: int = 5, sample_size: int = None) -> np.ndarray:
        """
        Fit DBSCAN clustering model
        
        Parameters:
        -----------
        eps : float, optional
            Epsilon parameter (max distance between points in same cluster)
            If None, uses previously calculated optimal value or 0.5
        min_samples : int
            MinPts parameter (minimum number of samples in a neighborhood)
        sample_size : int, optional
            Maximum number of samples to use for fitting (if None, use all data)
            
        Returns:
        --------
        labels : numpy array
            Cluster labels for each data point (-1 represents noise)
        """
        # Decide what data to use
        if self.pca_components is not None:
            X = self.pca_components[:, :3]  # Use first 3 PCA components
            print(f"Using PCA components for clustering")
        else:
            X = self.df[self.features].values
            print(f"Using original features for clustering")
        
        # Sample data if it's too large and sample_size is specified
        if sample_size is not None and len(X) > sample_size:
            print(f"Sampling {sample_size} records from {len(X)} for faster clustering...")
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            
            # Use optimal eps if not provided
            if eps is None:
                eps = self.clustering_results.get('optimal_eps', 0.5)
                print(f"Using eps={eps} for DBSCAN")
            
            # Fit DBSCAN on sample
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            sample_labels = dbscan.fit_predict(X_sample)
            
            # Predict on full dataset 
            # DBSCAN doesn't have a predict method, so we need to refit on full data
            print("Applying clustering to full dataset...")
            full_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = full_dbscan.fit_predict(X)
            
            # Store the full model
            dbscan = full_dbscan
        else:
            # Use optimal eps if not provided
            if eps is None:
                eps = self.clustering_results.get('optimal_eps', 0.5)
                print(f"Using eps={eps} for DBSCAN")
            
            # Fit DBSCAN on full dataset
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
        
        # Store results
        self.clustering_results['model'] = dbscan
        self.clustering_results['labels'] = labels
        self.clustering_results['eps'] = eps
        self.clustering_results['min_samples'] = min_samples
        
        # Report number of clusters and noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        return labels
    
    def plot_dbscan_clusters(self, figsize: tuple = (900, 700), sample_size: int = 10000) -> go.Figure:
        """
        Visualize DBSCAN clusters in 2D using the first two PCA components
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        sample_size : int
            Maximum number of points to display for better visualization performance
            
        Returns:
        --------
        fig : plotly Figure
            2D scatter plot of clusters
        """
        # Check if clustering has been performed
        if 'labels' not in self.clustering_results:
            raise ValueError("Run fit_dbscan() before plotting clusters")
        
        # Get labels
        labels = self.clustering_results['labels']
        
        # Decide what data to use for visualization
        if self.pca_components is not None:
            X = self.pca_components[:, :2]  # Use first 2 PCA components
            axis_labels = ['PCA 1', 'PCA 2']
        else:
            # Use the first two features for visualization
            X = self.df[self.features].iloc[:, :2].values
            axis_labels = self.features[:2]
        
        # Sample data if it's too large
        if len(X) > sample_size:
            print(f"Sampling {sample_size} points for visualization...")
            
            # Stratified sampling to ensure representation of all clusters
            unique_labels = np.unique(labels)
            indices = []
            
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                # Determine number of samples to take from this cluster
                if label == -1:  # Noise points - take fewer
                    n_samples = min(int(sample_size * 0.2), len(label_indices))
                else:
                    n_samples = min(int(sample_size * 0.8 / (len(unique_labels) - 1)), len(label_indices))
                
                if n_samples > 0:
                    sampled_indices = np.random.choice(label_indices, size=n_samples, replace=False)
                    indices.extend(sampled_indices)
            
            # If we still have room, add more samples randomly
            if len(indices) < sample_size:
                remaining = sample_size - len(indices)
                remaining_indices = list(set(range(len(X))) - set(indices))
                if remaining_indices:
                    extra_indices = np.random.choice(remaining_indices, size=min(remaining, len(remaining_indices)), replace=False)
                    indices.extend(extra_indices)
            
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each cluster and noise
        unique_labels = sorted(set(labels_sample))
        
        # First plot noise points (if any)
        if -1 in unique_labels:
            noise_mask = (labels_sample == -1)
            fig.add_trace(
                go.Scatter(
                    x=X_sample[noise_mask, 0],
                    y=X_sample[noise_mask, 1],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=self.get_cluster_color(-1),
                        symbol='circle',
                        opacity=0.5
                    ),
                    name=self.get_cluster_name(-1)
                )
            )
            unique_labels.remove(-1)  # Remove noise so we don't process it again
        
        # Then plot actual clusters
        for label in unique_labels:
            cluster_mask = (labels_sample == label)
            fig.add_trace(
                go.Scatter(
                    x=X_sample[cluster_mask, 0],
                    y=X_sample[cluster_mask, 1],
                    mode='markers',
                    marker=dict(
                        size=7,
                        color=self.get_cluster_color(label),
                        symbol='circle'
                    ),
                    name=self.get_cluster_name(label)
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f'DBSCAN Clustering Results (eps={self.clustering_results["eps"]:.3f}, min_samples={self.clustering_results["min_samples"]})',
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        return fig
    
    def plot_cluster_statistics(self, figsize: tuple = (900, 500)) -> go.Figure:
        """
        Create a visualization of cluster statistics (size, density, etc.)
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : plotly Figure
            Statistical visualization of clusters
        """
        # Check if clustering has been performed
        if 'labels' not in self.clustering_results:
            raise ValueError("Run fit_dbscan() before plotting statistics")
        
        # Get labels
        labels = self.clustering_results['labels']
        
        # Calculate statistics for each cluster
        unique_labels = sorted(set(labels))
        cluster_stats = []
        
        # Decide what data to use for statistics
        if self.pca_components is not None:
            X = self.pca_components[:, :3]  # Use first 3 PCA components
        else:
            X = self.df[self.features].values
        
        for label in unique_labels:
            # Skip noise for some calculations
            if label == -1:
                cluster_stats.append({
                    'cluster': self.get_cluster_name(label),
                    'size': np.sum(labels == label),
                    'percentage': np.mean(labels == label) * 100,
                    'color': self.get_cluster_color(label),
                    'is_noise': True
                })
            else:
                # Get points in this cluster
                cluster_points = X[labels == label]
                
                # Calculate statistics
                cluster_stats.append({
                    'cluster': self.get_cluster_name(label),
                    'size': len(cluster_points),
                    'percentage': (len(cluster_points) / len(labels)) * 100,
                    'color': self.get_cluster_color(label),
                    'is_noise': False
                })
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(cluster_stats)
        
        # Create figure
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Cluster Size', 'Cluster Percentage'),
                           specs=[[{"type": "bar"}, {"type": "pie"}]])
        
        # Add bar chart of cluster sizes
        fig.add_trace(
            go.Bar(
                x=stats_df['cluster'],
                y=stats_df['size'],
                marker_color=stats_df['color'],
                name='Cluster Size'
            ),
            row=1, col=1
        )
        
        # Add pie chart of percentages
        fig.add_trace(
            go.Pie(
                labels=stats_df['cluster'],
                values=stats_df['percentage'],
                marker=dict(colors=stats_df['color']),
                name='Cluster Percentage'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'DBSCAN Cluster Statistics (eps={self.clustering_results["eps"]:.3f}, min_samples={self.clustering_results["min_samples"]})',
            width=figsize[0],
            height=figsize[1],
            showlegend=False
        )
        
        return fig
    
    def parameter_sensitivity_analysis(self, 
                                      eps_range: List[float] = None, 
                                      min_samples_range: List[int] = None, 
                                      sample_size: int = 10000) -> go.Figure:
        """
        Analyze sensitivity of DBSCAN to different parameter combinations
        
        Parameters:
        -----------
        eps_range : list of float
            Range of epsilon values to test
        min_samples_range : list of int
            Range of min_samples values to test
        sample_size : int
            Number of samples to use for faster computation
            
        Returns:
        --------
        fig : plotly Figure
            Heatmap of cluster counts for different parameter combinations
        """
        # Set default parameter ranges if not provided
        if eps_range is None:
            optimal_eps = self.clustering_results.get('optimal_eps', 0.5)
            eps_range = np.linspace(optimal_eps * 0.5, optimal_eps * 1.5, 6)
            
        if min_samples_range is None:
            current_min_samples = self.clustering_results.get('min_samples', 5)
            min_samples_range = list(range(max(2, current_min_samples - 3), 
                                          current_min_samples + 4))
        
        # Decide what data to use
        if self.pca_components is not None:
            X = self.pca_components[:, :3]  # Use first 3 PCA components
            print(f"Using PCA components for sensitivity analysis")
        else:
            X = self.df[self.features].values
            print(f"Using original features for sensitivity analysis")
        
        # Sample data if it's too large
        if len(X) > sample_size:
            print(f"Sampling {sample_size} records for faster computation...")
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Initialize results grid
        results = []
        
        # Run DBSCAN with different parameter combinations
        for eps in tqdm(eps_range, desc="Testing eps values"):
            for min_samples in min_samples_range:
                # Fit DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_sample)
                
                # Count clusters and noise points
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_percentage = n_noise / len(labels) * 100
                
                # Calculate silhouette score if there are at least 2 clusters and not all points are noise
                silhouette = np.nan
                if n_clusters >= 2 and n_noise < len(labels):
                    try:
                        # Only use non-noise points for silhouette
                        mask = labels != -1
                        if np.sum(mask) > n_clusters:  # Ensure we have enough points
                            silhouette = silhouette_score(
                                X_sample[mask], 
                                labels[mask],
                                sample_size=min(1000, np.sum(mask))
                            )
                    except Exception as e:
                        print(f"Error calculating silhouette for eps={eps}, min_samples={min_samples}: {e}")
                
                # Store results
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_percentage': noise_percentage,
                    'silhouette': silhouette
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Number of Clusters',
                'Noise Percentage',
                'Silhouette Score',
                'Parameter Recommendation'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ]
        )
        
        # Helper function to pivot the data for heatmaps
        def create_heatmap_data(metric):
            pivoted = results_df.pivot(
                index='min_samples', 
                columns='eps', 
                values=metric
            )
            return pivoted.values, pivoted.index, pivoted.columns
        
        # Add number of clusters heatmap
        z_clusters, y_clusters, x_clusters = create_heatmap_data('n_clusters')
        fig.add_trace(
            go.Heatmap(
                z=z_clusters,
                x=x_clusters,
                y=y_clusters,
                colorscale='Viridis',
                colorbar=dict(title='Clusters'),
                name='Number of Clusters'
            ),
            row=1, col=1
        )
        
        # Add noise percentage heatmap
        z_noise, y_noise, x_noise = create_heatmap_data('noise_percentage')
        fig.add_trace(
            go.Heatmap(
                z=z_noise,
                x=x_noise,
                y=y_noise,
                colorscale='Reds',
                colorbar=dict(title='Noise %'),
                name='Noise Percentage'
            ),
            row=1, col=2
        )
        
        # Add silhouette score heatmap
        z_silhouette, y_silhouette, x_silhouette = create_heatmap_data('silhouette')
        fig.add_trace(
            go.Heatmap(
                z=z_silhouette,
                x=x_silhouette,
                y=y_silhouette,
                colorscale='Blues',
                colorbar=dict(title='Silhouette'),
                name='Silhouette Score'
            ),
            row=2, col=1
        )
        
        # Find optimal parameters based on silhouette score
        valid_results = results_df.dropna(subset=['silhouette'])
        if not valid_results.empty:
            # Find parameters with highest silhouette
            best_row = valid_results.loc[valid_results['silhouette'].idxmax()]
            best_eps = best_row['eps']
            best_min_samples = best_row['min_samples']
            best_silhouette = best_row['silhouette']
            best_clusters = best_row['n_clusters']
            best_noise = best_row['noise_percentage']
            
            # Create a scatter plot showing all valid parameter combinations
            fig.add_trace(
                go.Scatter(
                    x=valid_results['eps'],
                    y=valid_results['min_samples'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=valid_results['silhouette'],
                        colorscale='Blues',
                        showscale=False
                    ),
                    text=[f"eps={row['eps']:.3f}<br>min_samples={row['min_samples']}<br>"
                          f"silhouette={row['silhouette']:.3f}<br>"
                          f"clusters={row['n_clusters']}<br>"
                          f"noise={row['noise_percentage']:.1f}%" 
                          for _, row in valid_results.iterrows()],
                    hoverinfo='text',
                    name='Parameter Combinations'
                ),
                row=2, col=2
            )
            
            # Add a special marker for the best parameters
            fig.add_trace(
                go.Scatter(
                    x=[best_eps],
                    y=[best_min_samples],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='star',
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    name='Recommended Parameters',
                    text=[f"Recommended:<br>eps={best_eps:.3f}<br>min_samples={best_min_samples}<br>"
                          f"silhouette={best_silhouette:.3f}<br>"
                          f"clusters={best_clusters}<br>"
                          f"noise={best_noise:.1f}%"],
                    hoverinfo='text'
                ),
                row=2, col=2
            )
            
            # Store recommended parameters
            self.clustering_results['recommended_eps'] = best_eps
            self.clustering_results['recommended_min_samples'] = best_min_samples
        
        # Update layout and axis labels
        fig.update_layout(
            title='DBSCAN Parameter Sensitivity Analysis',
            width=1000,
            height=800,
            showlegend=False
        )
        
        # Update x and y axes for all subplots
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Epsilon (ε)" if i == 2 else None, row=i, col=j)
                fig.update_yaxes(title_text="Min Samples" if j == 1 else None, row=i, col=j)
        
        return fig
    
    def plot_silhouette(self, figsize: tuple = (900, 600), sample_size: int = 20000) -> go.Figure:
        """
        Create a silhouette plot to visualize cluster quality using Plotly.
        The height of each cluster section will accurately reflect the relative cluster size.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        sample_size : int
            Maximum number of samples to use for silhouette calculation
            
        Returns:
        --------
        fig : plotly Figure
            Silhouette plot visualization
        """
        # Check if clustering has been performed
        if 'labels' not in self.clustering_results:
            raise ValueError("Run fit_dbscan() before plotting silhouette")
        
        # Get labels
        labels = self.clustering_results['labels']
        
        # Remove noise points (labeled as -1) for silhouette calculation
        mask = labels != -1
        valid_labels = labels[mask]
        
        # Check if we have at least two clusters (excluding noise)
        n_clusters = len(set(valid_labels))
        if n_clusters < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Cannot calculate silhouette scores with fewer than 2 clusters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Silhouette Analysis for DBSCAN Clustering",
                width=figsize[0],
                height=figsize[1]
            )
            return fig
        
        # Decide what data to use
        if self.pca_components is not None:
            X = self.pca_components[:, :3][mask]  # Use first 3 PCA components
        else:
            X = self.df[self.features].values[mask]
        
        # Sample data if needed for faster silhouette calculation
        if len(X) > sample_size:
            print(f"Sampling {sample_size} records from {len(X)} for faster silhouette calculation...")
            
            # Stratified sampling to maintain cluster proportions
            unique_labels = sorted(set(valid_labels))
            indices = []
            cluster_proportions = {label: np.mean(valid_labels == label) for label in unique_labels}
            
            for label in unique_labels:
                # Find indices for this cluster
                cluster_indices = np.where(valid_labels == label)[0]
                
                # Calculate how many samples to take from this cluster
                n_samples = int(sample_size * cluster_proportions[label])
                if n_samples > 0:  # Ensure we take at least some samples
                    cluster_sample = np.random.choice(cluster_indices, 
                                                     size=min(n_samples, len(cluster_indices)), 
                                                     replace=False)
                    indices.extend(cluster_sample)
            
            # If we didn't get enough samples, add more randomly
            if len(indices) < sample_size:
                remaining = sample_size - len(indices)
                all_indices = set(range(len(X)))
                remaining_indices = list(all_indices - set(indices))
                if remaining_indices:
                    extra_indices = np.random.choice(remaining_indices, 
                                                   size=min(remaining, len(remaining_indices)), 
                                                   replace=False)
                    indices.extend(extra_indices)
            
            X_sample = X[indices]
            sample_labels = valid_labels[indices]
        else:
            X_sample = X
            sample_labels = valid_labels
        
        # Calculate silhouette scores
        silhouette_vals = silhouette_samples(X_sample, sample_labels)
        
        # Calculate average silhouette score
        avg_score = np.mean(silhouette_vals)
        
        # Create a DataFrame for visualization
        silhouette_df = pd.DataFrame({
            'sample_idx': range(len(silhouette_vals)),
            'cluster': sample_labels,
            'silhouette_val': silhouette_vals
        })
        
        # Sort within each cluster for better visualization
        silhouette_df = silhouette_df.sort_values(['cluster', 'silhouette_val'])
        
        # Create figure
        fig = go.Figure()
        
        # Add silhouette traces for each cluster
        # Scale the total height based on figure size
        total_height = figsize[1] * 0.8  # 80% of figure height for the plots
        
        # Starting position
        y_lower = 10
        
        # Calculate cluster counts and proportions for the full dataset (excluding noise)
        full_cluster_counts = {label: np.sum(valid_labels == label) for label in sorted(set(valid_labels))}
        total_non_noise = sum(full_cluster_counts.values())
        full_cluster_proportions = {label: count / total_non_noise for label, count in full_cluster_counts.items()}
        
        for cluster_id in sorted(set(sample_labels)):
            # Get silhouette values for current cluster
            cluster_silhouette_vals = silhouette_df[silhouette_df['cluster'] == cluster_id]['silhouette_val']
            cluster_silhouette_vals = cluster_silhouette_vals.sort_values()
            
            if len(cluster_silhouette_vals) == 0:
                continue  # Skip empty clusters
                
            # Calculate height based on proportion
            cluster_height = total_height * full_cluster_proportions[cluster_id]
            
            # Calculate y positions
            y_upper = y_lower + cluster_height
            y_positions = np.linspace(y_lower, y_upper - 1, len(cluster_silhouette_vals))
            
            # Use consistent colors
            fill_color = self.get_cluster_color(cluster_id, transparent=True)
            line_color = self.get_cluster_color(cluster_id)
            
            # Add the silhouette plot for this cluster
            fig.add_trace(
                go.Scatter(
                    x=cluster_silhouette_vals,
                    y=y_positions,
                    mode='lines',
                    line=dict(width=0.5, color=line_color),
                    fill='tozerox',
                    fillcolor=fill_color,
                    name=f"{self.get_cluster_name(cluster_id)} ({full_cluster_counts[cluster_id]} samples, {full_cluster_proportions[cluster_id]:.1%})"
                )
            )
            
            # Update y_lower for next cluster
            y_lower = y_upper + 5
        
        # Add a vertical line for the average silhouette score
        fig.add_vline(
            x=avg_score, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Avg Silhouette: {avg_score:.3f}",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title=f'Silhouette Analysis for DBSCAN Clustering (eps={self.clustering_results["eps"]:.3f}, min_samples={self.clustering_results["min_samples"]})',
            xaxis_title='Silhouette Coefficient',
            yaxis_title='Cluster Distribution',
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            xaxis=dict(range=[-0.1, 1.05]),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def plot_intercluster_distance(self, figsize: tuple = (900, 700)) -> go.Figure:
        """
        Create a circle-based visualization showing relationships between DBSCAN cluster cores.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : plotly Figure
            Visualization of cluster relationships
        """
        from scipy.spatial.distance import pdist, squareform
        from sklearn.manifold import MDS
        
        # Check if clustering has been performed
        if 'labels' not in self.clustering_results:
            raise ValueError("Run fit_dbscan() before plotting intercluster distance")
        
        # Get labels
        labels = self.clustering_results['labels']
        
        # Get unique clusters (excluding noise)
        unique_clusters = sorted([label for label in set(labels) if label != -1])
        
        # If less than 2 clusters (excluding noise), show a message
        if len(unique_clusters) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Cannot visualize distances with fewer than 2 clusters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Intercluster Distance Visualization",
                width=figsize[0],
                height=figsize[1]
            )
            return fig
        
        # Decide what data to use
        if self.pca_components is not None:
            X = self.pca_components[:, :3]  # Use first 3 PCA components
        else:
            X = self.df[self.features].values
        
        # Calculate cluster representatives (mean of each cluster)
        representatives = []
        cluster_sizes = []
        
        for cluster_id in unique_clusters:
            cluster_points = X[labels == cluster_id]
            representatives.append(np.mean(cluster_points, axis=0))
            cluster_sizes.append(len(cluster_points))
        
        representatives = np.array(representatives)
        
        # Compute pairwise distances between representatives
        distances = pdist(representatives)
        distance_matrix = squareform(distances)
        
        # Use MDS to position clusters in 2D space based on their distances
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        positions = mds.fit_transform(distance_matrix)
        
        # Scale sizes based on cluster sizes
        max_size = max(cluster_sizes)
        size_scale = 100  # Max circle size
        sizes = [size / max_size * size_scale for size in cluster_sizes]
        
        # Create figure
        fig = go.Figure()
        
        # Add circles for each cluster
        for i, cluster_id in enumerate(unique_clusters):
            fig.add_trace(go.Scatter(
                x=[positions[i, 0]],
                y=[positions[i, 1]],
                mode='markers',
                marker=dict(
                    size=sizes[i],
                    color=self.get_cluster_color(cluster_id),
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                name=self.get_cluster_name(cluster_id),
                text=[f"{self.get_cluster_name(cluster_id)}<br>{cluster_sizes[i]} points"],
                hoverinfo='text'
            ))
        
        # Add lines between clusters with distance labels
        max_dist = np.max(distances)
        for i in range(len(unique_clusters)):
            for j in range(i+1, len(unique_clusters)):
                # Calculate line opacity based on inverse of distance
                opacity = 0.8 * (1 - distance_matrix[i, j] / max_dist) + 0.2
                
                # Add a line connecting the clusters
                fig.add_trace(go.Scatter(
                    x=[positions[i, 0], positions[j, 0]],
                    y=[positions[i, 1], positions[j, 1]],
                    mode='lines',
                    line=dict(
                        width=1,
                        color=f'rgba(100, 100, 100, {opacity:.2f})'
                    ),
                    hoverinfo='text',
                    text=f"Distance: {distance_matrix[i, j]:.3f}",
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=f"DBSCAN Intercluster Relationship Visualization (eps={self.clustering_results['eps']:.3f})",
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            hovermode='closest',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        return fig
    
    def plot_feature_importance(self, n_repeats: int = 3, sample_size: int = 10000) -> go.Figure:
        """
        Calculate and visualize feature importance for DBSCAN clustering using permutation importance.
        
        Parameters:
        -----------
        n_repeats : int
            Number of times to permute a feature (lower means faster calculation)
        sample_size : int
            Maximum number of samples to use for faster computation
            
        Returns:
        --------
        fig : plotly Figure
            Feature importance bar chart
        """
        # Check if clustering has been performed
        if 'labels' not in self.clustering_results:
            raise ValueError("Run fit_dbscan() before calculating feature importance")
        
        # Get labels
        labels = self.clustering_results['labels']
        
        # Get parameters from the existing model
        eps = self.clustering_results['eps']
        min_samples = self.clustering_results['min_samples']
        
        # Always define X_orig first to ensure it's available in all code paths
        X_orig = self.df[self.features].values
        
        # Sample data if it's too large (for faster computation)
        if len(X_orig) > sample_size:
            print(f"Sampling {sample_size} records from {len(X_orig)} for faster calculation...")
            indices = np.random.choice(len(X_orig), sample_size, replace=False)
            X_sample = X_orig[indices]
            sample_labels = labels[indices]
        else:
            X_sample = X_orig
            sample_labels = labels
        
        # Create a scorer for DBSCAN
        # For DBSCAN, we use silhouette score to measure quality
        # Note: We need to remove noise points
        def dbscan_scorer(estimator, X, y=None):
            # Predict clusters
            pred_labels = estimator.fit_predict(X)
            
            # If all points are noise or there's only one cluster, return 0
            unique_clusters = set(pred_labels)
            if len(unique_clusters) <= 1 or (len(unique_clusters) == 2 and -1 in unique_clusters):
                return 0
            
            # Remove noise points for silhouette calculation
            mask = pred_labels != -1
            if np.sum(mask) <= 1:
                return 0
                
            # Calculate silhouette score using non-noise points
            try:
                return silhouette_score(
                    X[mask], 
                    pred_labels[mask],
                    sample_size=min(1000, np.sum(mask))
                )
            except:
                return 0
        
        # Initialize DBSCAN with the same parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Calculate permutation importance
        print(f"Calculating feature importance with {n_repeats} repeats (this may take a moment)...")
        try:
            result = permutation_importance(
                dbscan, 
                X_sample,
                None,  # No target for unsupervised
                scoring=dbscan_scorer,
                n_repeats=n_repeats,
                random_state=42
            )
            print("Feature importance calculation complete!")
        except Exception as e:
            print(f"Error in feature importance calculation: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Could not calculate feature importance: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Feature Importance for DBSCAN Clustering",
                width=900,
                height=500
            )
            return fig
        
        # Create DataFrame of results
        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': result.importances_mean,
            'StdDev': result.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            error_x='StdDev',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title=f'Feature Importance for DBSCAN Clustering (eps={eps:.3f}, min_samples={min_samples})',
            xaxis_title='Silhouette Score Reduction (Higher = More Important)',
            yaxis_title='Feature',
            width=900,
            height=max(500, 20 * len(self.features)),
            coloraxis_showscale=False
        )
        
        return fig
