import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from tqdm.notebook import tqdm
from sklearn.inspection import permutation_importance
import os

class KMeansClusterAnalysis:
    def __init__(self, df: pd.DataFrame, features: List[str] = None, transformer = None, pca_components: np.ndarray = None, n_jobs: int = 4):
        """
        Initialize K-means clustering analysis
        
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
        n_jobs : int
            Number of parallel jobs for computation
        """
        self.df = df
        self.transformer = transformer
        self.n_jobs = n_jobs
        
        # Auto-detect numeric features if not specified
        if features is None:
            self.features = df.select_dtypes(include=['number']).columns.tolist()
        else:
            self.features = features
            
        # Store PCA components if provided
        self.pca_components = pca_components
        
        # Results storage
        self.kmeans_results = {}
        
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
        return f'Cluster {cluster_idx}'
    
    def get_cluster_color(self, cluster_idx: int, transparent: bool = False) -> str:
        """Return a consistent color for a cluster index"""
        color_list = self.cluster_colors_transparent if transparent else self.cluster_colors
        return color_list[cluster_idx % len(color_list)]
        
    def elbow_method(self, k_range: range) -> Dict:
        """Optimized elbow method using MiniBatchKMeans."""
        inertias = []
        silhouette_scores = []
        
        # Decide what data to use for clustering
        if self.pca_components is not None:
            X = self.pca_components[:, :3]  # Use first 3 PCA components
            print(f"Using PCA components for elbow analysis")
        else:
            X = self.df[self.features].values
            print(f"Using original features for elbow analysis")
        
        for k in tqdm(k_range, desc="Computing clusters"):
            kmeans = MiniBatchKMeans(
                n_clusters=k, 
                batch_size=1024,
                random_state=42
            )
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            if k > 1:
                score = silhouette_score(
                    X, 
                    kmeans.labels_,
                    sample_size=min(20000, len(X))
                )
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)  # Can't compute silhouette for k=1
            
        return {
            'k_range': list(k_range),
            'inertia': inertias,
            'silhouette': silhouette_scores
        }
    
    def plot_elbow(self, k_range: range) -> go.Figure:
        """Plot elbow curve with plotly."""
        results = self.elbow_method(k_range)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Use consistent colors from the palette
        fig.add_trace(
            go.Scatter(x=results['k_range'], y=results['inertia'], 
                        name="Inertia", line=dict(color=self.cluster_colors[0]))
        )
        
        fig.add_trace(
            go.Scatter(x=results['k_range'][1:], y=results['silhouette'][1:],
                        name="Silhouette Score", line=dict(color=self.cluster_colors[1])),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Elbow Method with Silhouette Score',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Inertia',
            yaxis2_title='Silhouette Score',
            width=900,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def fit_kmeans(self, n_clusters: int, sample_size: int = None) -> np.ndarray:
        """
        Fit K-means clustering model
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        sample_size : int, optional
            Maximum number of samples to use for faster computation
            
        Returns:
        --------
        labels : numpy array
            Cluster labels for each data point
        """
        # For visualization - use PCA components if available
        if self.pca_components is not None:
            # Use PCA components for visualization model
            X_viz = self.pca_components[:, :3]  # Use only the first 3 components
            
            # Sample if needed
            if sample_size is not None and len(X_viz) > sample_size:
                print(f"Sampling {sample_size} records for PCA-based clustering...")
                indices = np.random.choice(len(X_viz), sample_size, replace=False)
                X_viz_sample = X_viz[indices]
                kmeans_pca = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
                kmeans_pca.fit(X_viz_sample)
            else:
                kmeans_pca = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
                kmeans_pca.fit(X_viz)
                
            # Generate labels for all data
            viz_labels = kmeans_pca.predict(X_viz)
            self.kmeans_results['pca_model'] = kmeans_pca
            self.kmeans_results['pca_labels'] = viz_labels
    
        # For feature importance - always use original features
        X_orig = self.df[self.features].values
        
        # Sample if needed
        if sample_size is not None and len(X_orig) > sample_size:
            print(f"Sampling {sample_size} records for feature-based clustering...")
            indices = np.random.choice(len(X_orig), sample_size, replace=False)
            X_orig_sample = X_orig[indices]
            kmeans_orig = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
            kmeans_orig.fit(X_orig_sample)
        else:
            kmeans_orig = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
            kmeans_orig.fit(X_orig)
        
        # Generate labels for all data
        orig_labels = kmeans_orig.predict(X_orig)
        
        self.kmeans_results['orig_model'] = kmeans_orig
        self.kmeans_results['labels'] = orig_labels
        
        return orig_labels
    
    def plot_feature_importance(self, n_clusters: int = None, n_repeats: int = 3, sample_size: int = 100000, top_n: int = 10) -> go.Figure:
        """
        Calculate and visualize feature importance for K-Means clustering using permutation importance.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters for K-Means (if None, uses previously fitted model)
        n_repeats : int
            Number of times to permute a feature (lower means faster calculation)
        sample_size : int
            Maximum number of samples to use for faster computation
        top_n : int
            Number of top features to display
            
        Returns:
        --------
        fig : plotly Figure
            Feature importance bar chart
        """
        # Always define X_orig first to ensure it's available in all code paths
        X_orig = self.df[self.features].values
        
        # Sample data if it's too large (for faster computation)
        if len(X_orig) > sample_size:
            print(f"Sampling {sample_size} records from {len(X_orig)} for faster calculation...")
            indices = np.random.choice(len(X_orig), sample_size, replace=False)
            X_sample = X_orig[indices]
        else:
            X_sample = X_orig
        
        # Use the existing model if available
        if n_clusters is None and 'orig_model' in self.kmeans_results:
            kmeans_orig = self.kmeans_results['orig_model']
            # Fix: Update n_clusters to the actual number in the model
            n_clusters = kmeans_orig.n_clusters
        else:
            # Create a new model if needed
            if n_clusters is None:
                n_clusters = self.kmeans_results.get('orig_model', {}).n_clusters if 'orig_model' in self.kmeans_results else 4
            
            # Train a KMeans model directly on original features for importance calculation
            kmeans_orig = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=1024,
                random_state=42
            )
            
            print("Fitting MiniBatchKMeans on original features...")
            kmeans_orig.fit(X_orig)
        
        # Define scorer function for silhouette - must accept y even though we don't use it
        def silhouette_scorer(estimator, X, y=None):
            labels = estimator.predict(X)
            # Use much smaller sample for silhouette score calculation
            return silhouette_score(X, labels, sample_size=min(1000, len(X)))
        
        # Calculate permutation importance on original features
        print(f"Calculating feature importance with {n_repeats} repeats (this may take a moment)...")
        result = permutation_importance(
            kmeans_orig, 
            X_sample,  # Use the sampled dataset
            None,  # No target for unsupervised
            scoring=silhouette_scorer,
            n_repeats=n_repeats,
            random_state=42
        )
        print("Feature importance calculation complete!")
        
        # Create DataFrame of results
        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': result.importances_mean,
            'StdDev': result.importances_std
        })
        
        # Sort by importance and take top N
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create bar chart with consistent colors
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
            title=f'Top {top_n} Feature Importance for K-Means Clustering (k={n_clusters})',
            xaxis_title='Silhouette Score Reduction (Higher = More Important)',
            yaxis_title='Feature',
            width=900,
            height=max(500, 25 * len(importance_df)),
            coloraxis_showscale=False,
            yaxis={'categoryorder':'total ascending'} # Ensure correct order
        )
        
        return fig
    
    def plot_silhouette(self, n_clusters: int = None, figsize: tuple = (900, 600), sample_size: int = 20000) -> go.Figure:
        """
        Create a silhouette plot to visualize cluster quality using Plotly.
        The height of each cluster section will accurately reflect the relative cluster size.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters (if None, uses previously fitted model)
        figsize : tuple
            Figure size (width, height)
        sample_size : int
            Maximum number of samples to use for silhouette calculation
            
        Returns:
        --------
        fig : plotly Figure
            Silhouette plot visualization
        """
        # Get number of clusters
        if n_clusters is None:
            if 'orig_model' in self.kmeans_results:
                n_clusters = self.kmeans_results['orig_model'].n_clusters
            else:
                n_clusters = 3  # Default
        
        # Get actual cluster sizes from the complete dataset
        if 'labels' in self.kmeans_results and len(self.kmeans_results['labels']) == len(self.df):
            # Use existing labels if they match the dataset size
            full_labels = self.kmeans_results['labels']
            full_cluster_counts = np.bincount(full_labels)
            full_cluster_proportions = full_cluster_counts / full_cluster_counts.sum()
        else:
            # Otherwise, run on the full dataset to get true proportions
            print("Computing cluster proportions on full dataset...")
            kmeans_full = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
            full_labels = kmeans_full.fit_predict(self.df[self.features].values)
            full_cluster_counts = np.bincount(full_labels)
            full_cluster_proportions = full_cluster_counts / full_cluster_counts.sum()
        
        # Sample data if needed for faster silhouette calculation
        X_sample = self.df[self.features].values
        if len(X_sample) > 20000:
            print(f"Sampling 20,000 records from {len(X_sample)} for faster silhouette calculation...")
            
            # Stratified sampling to maintain cluster proportions
            sample_size = 20000
            indices = []
            
            for i in range(n_clusters):
                # Find indices for this cluster
                cluster_indices = np.where(full_labels == i)[0]
                
                # Calculate how many samples to take from this cluster
                # to maintain the same proportion as in the full dataset
                n_samples = int(sample_size * full_cluster_proportions[i])
                if n_samples > 0:  # Ensure we take at least some samples
                    cluster_sample = np.random.choice(cluster_indices, 
                                                      size=min(n_samples, len(cluster_indices)), 
                                                      replace=False)
                    indices.extend(cluster_sample)
            
            # If we didn't get enough samples (due to rounding), add more randomly
            if len(indices) < sample_size:
                remaining = sample_size - len(indices)
                all_indices = set(range(len(X_sample)))
                remaining_indices = list(all_indices - set(indices))
                if remaining_indices:
                    extra_indices = np.random.choice(remaining_indices, 
                                                   size=min(remaining, len(remaining_indices)), 
                                                   replace=False)
                    indices.extend(extra_indices)
            
            X_sample = X_sample[indices]
            sample_labels = full_labels[indices]
        else:
            sample_labels = full_labels
        
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
        
        for i in range(n_clusters):
            # Get silhouette values for current cluster
            cluster_silhouette_vals = silhouette_df[silhouette_df['cluster'] == i]['silhouette_val']
            cluster_silhouette_vals = cluster_silhouette_vals.sort_values()
            
            if len(cluster_silhouette_vals) == 0:
                continue  # Skip empty clusters
                
            # Calculate height based on proportion
            cluster_height = total_height * full_cluster_proportions[i]
            
            # Calculate y positions
            y_upper = y_lower + cluster_height
            y_positions = np.linspace(y_lower, y_upper - 1, len(cluster_silhouette_vals))
            
            # Use consistent colors
            fill_color = self.get_cluster_color(i, transparent=True)
            line_color = self.get_cluster_color(i)
            
            # Add the silhouette plot for this cluster
            fig.add_trace(
                go.Scatter(
                    x=cluster_silhouette_vals,
                    y=y_positions,
                    mode='lines',
                    line=dict(width=0.5, color=line_color),
                    fill='tozerox',
                    fillcolor=fill_color,
                    name=f"{self.get_cluster_name(i)} ({full_cluster_counts[i]} samples, {full_cluster_proportions[i]:.1%})"
                )
            )
            
            # Update y_lower for next cluster
            y_lower = y_upper + 5  # Less spacing between clusters
        
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
            title=f'Silhouette Analysis for KMeans Clustering (k={n_clusters})',
            xaxis_title='Silhouette Coefficient',
            yaxis_title='Cluster Distribution',
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            xaxis=dict(range=[-0.1, 1.05]),  # Silhouette values range from -1 to 1
            yaxis=dict(showticklabels=False)  # Hide y-axis tick labels
        )
        
        return fig
    
    def plot_intercluster_distance(self, n_clusters: int = None, figsize: tuple = (900, 700)) -> go.Figure:
        """
        Create a circle-based visualization showing relationships between cluster centers.
        """
        from scipy.spatial.distance import pdist, squareform
        from sklearn.manifold import MDS
        
        # Get number of clusters
        if n_clusters is None:
            if 'orig_model' in self.kmeans_results:
                n_clusters = self.kmeans_results['orig_model'].n_clusters
            else:
                n_clusters = 3  # Default
        
        # Get cluster centers and sizes
        if 'orig_model' in self.kmeans_results and (n_clusters is None or 
                n_clusters == self.kmeans_results['orig_model'].n_clusters):
            centers = self.kmeans_results['orig_model'].cluster_centers_
            labels = self.kmeans_results.get('labels', None)
        else:
            # Create and fit a new model
            print(f"Fitting KMeans with {n_clusters} clusters...")
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
            X = self.df[self.features].values
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
        
        # Get cluster sizes if labels are available
        if labels is not None:
            cluster_sizes = np.bincount(labels)
            size_scale = 100  # Max circle size
            sizes = (cluster_sizes / cluster_sizes.max()) * size_scale
        else:
            sizes = np.ones(n_clusters) * 50  # Default size if no labels
        
        # Compute pairwise distances between centers
        distances = pdist(centers)
        distance_matrix = squareform(distances)
        
        # Use MDS to position clusters in 2D space based on their distances
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        positions = mds.fit_transform(distance_matrix)
        
        # Create figure
        fig = go.Figure()
        
        # Add circles for each cluster
        for i in range(n_clusters):
            fig.add_trace(go.Scatter(
                x=[positions[i, 0]],
                y=[positions[i, 1]],
                mode='markers',
                marker=dict(
                    size=sizes[i],
                    color=self.get_cluster_color(i),
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                name=self.get_cluster_name(i),
                text=[self.get_cluster_name(i)],
                hoverinfo='text'
            ))
        
        # Add lines between clusters with distance labels
        max_dist = np.max(distances)
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                dist = distance_matrix[i, j]
                normalized_dist = dist/max_dist
                
                # Calculate line points including the midpoint for text placement
                line_color = self.get_cluster_color(i)
                # Width based on proximity - thicker for closer clusters
                width = 1.5 + 3 * (1 - normalized_dist)
                
                # Add a line with distance shown in the middle
                fig.add_trace(go.Scatter(
                    x=[positions[i, 0], positions[j, 0]],
                    y=[positions[i, 1], positions[j, 1]],
                    mode='lines',
                    line=dict(
                        color=line_color,
                        width=width,
                        dash='solid' if normalized_dist < 0.5 else 'dot'  # Solid for close, dotted for far
                    ),
                    hovertext=[f'Distance: {dist:.2f}'],
                    showlegend=False
                ))
                
                # Add text at the midpoint
                mid_x = (positions[i, 0] + positions[j, 0]) / 2
                mid_y = (positions[i, 1] + positions[j, 1]) / 2
                
                fig.add_trace(go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode='text',
                    text=[f"{dist:.2f}"],
                    textposition='middle center',
                    textfont=dict(
                        color=line_color,
                        size=10,
                        family='Arial'
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title="Intercluster Relationship Visualization",
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
    
    def plot_clusters_3d(self, n_clusters: int = None) -> go.Figure:
        """
        Plot 3D scatter plot of clusters using PCA components (if available).
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters (uses previously fitted model if None)
            
        Returns:
        --------
        fig : plotly Figure
            3D scatter plot of clusters
        """
        # Check if we have PCA components
        if self.pca_components is None or self.pca_components.shape[1] < 3:
            raise ValueError("PCA components not available or fewer than 3 dimensions")
        
        # Get or compute labels
        if n_clusters is None and 'pca_labels' in self.kmeans_results:
            labels = self.kmeans_results['pca_labels']
        else:
            # Fit model if needed
            if n_clusters is None:
                n_clusters = 4  # Default
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
            labels = kmeans.fit_predict(self.pca_components[:, :3])
        
        # Create a discrete color map based on our cluster colors
        cluster_colors_dict = {}
        for i in range(max(labels) + 1):
            cluster_colors_dict[i] = self.get_cluster_color(i)
            
        # Map labels to colors
        marker_colors = [cluster_colors_dict.get(l, '#777777') for l in labels]
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=self.pca_components[:, 0],  # First PCA component
                y=self.pca_components[:, 1],  # Second PCA component
                z=self.pca_components[:, 2],  # Third PCA component
                mode='markers',
                marker=dict(
                    size=5,
                    color=marker_colors,
                ),
                text=[f'{self.get_cluster_name(l)}' for l in labels]
            )
        ])
        
        fig.update_layout(
            title='3D Cluster Visualization (PCA Components)',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            width=900,
            height=900
        )
        
        return fig
