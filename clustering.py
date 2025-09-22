import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, Any
import warnings
import faiss
warnings.filterwarnings('ignore')

from EmbeddingContext import LocalEmbeddingContext


class EmbeddingClusterAnalyzer:
    """
    Clustering analysis tools for embedding databases including k-means clustering and UMAP dimensionality reduction.
    """
    
    def __init__(self, context: LocalEmbeddingContext):
        """
        Initialize the analyzer with an embedding context.
        
        Args:
            context: LocalEmbeddingContext with built index
        """
        self.context = context
        self.embeddings = None
        self.metadata = None
        self.clusters = None
        self.umap_embeddings = None
        self.cluster_labels = None
        self.n_clusters = None
        
    def extract_embeddings(self) -> bool:
        """
        Extract embeddings and metadata directly from the context's FAISS index.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.context is None or self.context.index is None:
            print("❌ No index available. Please build an index first.")
            return False
            
        try:
            # Get embeddings from FAISS index
            if hasattr(self.context.backend, '_deduper') and self.context.backend._deduper is not None:
                # Extract all vectors from the FAISS index
                n_vectors = self.context.backend.count()
                if n_vectors == 0:
                    print("❌ No vectors in index.")
                    return False
                    
                # Get dimension from index
                dim = self.context.index.d
                
                print(f"🔍 Extracting {n_vectors} vectors directly from FAISS index...")
                
                # Extract vectors directly from FAISS IndexHNSWFlat using reconstruct
                try:
                    self.embeddings = np.zeros((n_vectors, dim), dtype='float32')
                    for i in range(n_vectors):
                        vec = self.context.index.reconstruct(i)
                        self.embeddings[i] = vec
                    
                    print(f"✅ Successfully extracted vectors using FAISS reconstruct method")
                    
                except Exception as reconstruct_error:
                    print(f"⚠️ FAISS reconstruct failed: {reconstruct_error}")
                    print("🔄 Falling back to re-embedding...")
                    return self._fallback_reembedding()
                
                # Get metadata directly from backend
                if hasattr(self.context.backend, '_metas'):
                    all_metadata = self.context.backend._metas.copy()
                    self.metadata = pd.DataFrame(all_metadata)
                    
                    print(f"✅ Extracted {len(self.embeddings)} embeddings directly from FAISS ({self.embeddings.shape[1]}D)")
                    return True
                else:
                    print("❌ No metadata available in backend.")
                    return False
            else:
                print("❌ Backend deduper not available.")
                return False
                
        except Exception as e:
            print(f"⚠️ Error extracting from FAISS: {str(e)}")
            print("🔄 Falling back to re-embedding...")
            return self._fallback_reembedding()
    
    def _fallback_reembedding(self) -> bool:
        """
        Fallback method that re-embeds texts when direct extraction fails.
        """
        try:
            if not hasattr(self.context.backend, '_metas'):
                return False
                
            all_metadata = self.context.backend._metas.copy()
            
            print("🔄 Re-embedding texts for analysis...")
            texts = []
            for meta in all_metadata:
                title = meta.get('title', '')
                text = meta.get('text', '')
                doc_text = self.context.doc_prompt(title, text)
                texts.append(doc_text)
            
            # Re-embed in batches
            batch_size = 32
            embeddings_list = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_emb = self.context.embedding_model.encode(
                    batch,
                    batch_size=min(batch_size, len(batch)),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).astype('float32')
                embeddings_list.append(batch_emb)
            
            self.embeddings = np.vstack(embeddings_list)
            self.metadata = pd.DataFrame(all_metadata)
            
            print(f"✅ Re-embedded {len(self.embeddings)} embeddings with {self.embeddings.shape[1]} dimensions")
            return True
            
        except Exception as e:
            print(f"❌ Error in fallback re-embedding: {str(e)}")
            return False
    
    def find_optimal_clusters(self, max_clusters: int = 10, min_clusters: int = 2) -> int:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            max_clusters: Maximum number of clusters to try
            min_clusters: Minimum number of clusters to try
            
        Returns:
            int: Optimal number of clusters
        """
        if self.embeddings is None:
            if not self.extract_embeddings():
                return 5  # default fallback
        
        best_score = -1
        best_k = min_clusters
        
        print("🔍 Finding optimal number of clusters...")
        
        # Limit max clusters based on data size
        n_samples = len(self.embeddings)
        max_clusters = min(max_clusters, n_samples // 2)
        
        for k in range(min_clusters, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.embeddings)
                score = silhouette_score(self.embeddings, labels)
                print(f"  k={k}: silhouette_score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception as e:
                print(f"  k={k}: error - {str(e)}")
                continue
        
        print(f"✅ Optimal clusters: {best_k} (silhouette score: {best_score:.3f})")
        return best_k
    
    def perform_clustering(self, n_clusters: Optional[int] = None) -> bool:
        """
        Perform k-means clustering on the embeddings.
        
        Args:
            n_clusters: Number of clusters. If None, will find optimal number.
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.embeddings is None:
            if not self.extract_embeddings():
                return False
        
        try:
            if n_clusters is None:
                n_clusters = self.find_optimal_clusters()
            
            print(f"🎯 Performing k-means clustering with {n_clusters} clusters...")
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(self.embeddings)
            self.n_clusters = n_clusters
            
            # Store cluster centers for analysis
            self.cluster_centers = kmeans.cluster_centers_
            
            # Add cluster labels to metadata
            if self.metadata is not None:
                self.metadata = self.metadata.copy()
                self.metadata['cluster'] = self.cluster_labels
            
            # Print cluster statistics
            unique, counts = np.unique(self.cluster_labels, return_counts=True)
            print("📊 Cluster distribution:")
            for cluster_id, count in zip(unique, counts):
                percentage = (count / len(self.cluster_labels)) * 100
                print(f"  Cluster {cluster_id}: {count} items ({percentage:.1f}%)")
            
            print("✅ Clustering completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error in clustering: {str(e)}")
            return False
    
    def perform_umap_reduction(self, n_neighbors: int = 15, min_dist: float = 0.1, 
                              n_components: int = 2, random_state: int = 42) -> bool:
        """
        Perform UMAP dimensionality reduction.
        
        Args:
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            n_components: Number of dimensions to reduce to (usually 2 for visualization)
            random_state: Random state for reproducibility
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.embeddings is None:
            if not self.extract_embeddings():
                return False
        
        try:
            print("🗺️ Performing UMAP dimensionality reduction...")
            
            # Create UMAP reducer
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=random_state,
                verbose=False
            )
            
            # Fit and transform the embeddings
            self.umap_embeddings = reducer.fit_transform(self.embeddings)
            
            print(f"✅ UMAP reduction completed: {self.embeddings.shape[1]}D → {self.umap_embeddings.shape[1]}D")
            return True
            
        except Exception as e:
            print(f"❌ Error in UMAP reduction: {str(e)}")
            return False
    
    def analyze_clusters(self) -> Dict[str, Any]:
        """
        Analyze cluster characteristics and return summary statistics.
        
        Returns:
            dict: Analysis results including cluster summaries
        """
        if self.cluster_labels is None or self.metadata is None:
            return {"error": "No clustering results available"}
        
        analysis = {
            "n_clusters": self.n_clusters,
            "total_items": len(self.cluster_labels),
            "clusters": []
        }
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_data = self.metadata[cluster_mask]
            
            # Get most common words in titles and texts
            titles = cluster_data['title'].fillna('').str.lower()
            texts = cluster_data['text'].fillna('').str.lower()
            
            # Simple word frequency analysis
            all_text = ' '.join(titles) + ' ' + ' '.join(texts)
            words = all_text.split()
            from collections import Counter
            word_freq = Counter(words)
            
            # Filter out common stop words and get top terms
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            filtered_words = {word: count for word, count in word_freq.items() 
                            if len(word) > 2 and word not in stop_words}
            top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]
            
            cluster_info = {
                "id": int(cluster_id),
                "size": int(cluster_mask.sum()),
                "percentage": float((cluster_mask.sum() / len(self.cluster_labels)) * 100),
                "top_words": top_words,
                "sample_titles": cluster_data['title'].dropna().head(5).tolist()
            }
            
            analysis["clusters"].append(cluster_info)
        
        return analysis
    
    def create_scatter_plot(self, color_by: str = 'cluster', title: str = "UMAP Visualization") -> go.Figure:
        """
        Create an interactive scatter plot of the UMAP embeddings.
        
        Args:
            color_by: What to color points by ('cluster' or other metadata column)
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive scatter plot
        """
        if self.umap_embeddings is None:
            # Try to perform UMAP if not done yet
            if not self.perform_umap_reduction():
                return go.Figure().add_annotation(text="UMAP embeddings not available", 
                                                xref="paper", yref="paper", x=0.5, y=0.5)
        
        if self.metadata is None:
            return go.Figure().add_annotation(text="Metadata not available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for plotting
        plot_data = self.metadata.copy()
        plot_data['x'] = self.umap_embeddings[:, 0]
        plot_data['y'] = self.umap_embeddings[:, 1]
        
        # Prepare clean hover data - just title and cluster info
        plot_data['title_clean'] = plot_data['title'].fillna('Untitled').str[:80]
        plot_data['date_clean'] = plot_data.get('publish_date', 'N/A').astype(str).str[:10]
        
        # Create simple hover template - no text preview to avoid clipping
        hover_template = (
            "<b>%{customdata[0]}</b><br>" +
            "Cluster: %{customdata[1]}<br>" +
            "Date: %{customdata[2]}<br>" +
            "<extra></extra>"
        )
        
        # Prepare custom data for hover - only essential info
        custom_data = np.column_stack([
            plot_data['title_clean'],
            plot_data.get('cluster', 'N/A'),
            plot_data['date_clean']
        ])
        
        # Create the scatter plot
        if color_by == 'cluster' and 'cluster' in plot_data.columns:
            # Color by cluster with better, more readable colors
            # Avoid yellow and light colors that are hard to see
            readable_colors = [
                '#1f77b4',  # blue
                '#ff7f0e',  # orange  
                '#2ca02c',  # green
                '#d62728',  # red
                '#9467bd',  # purple
                '#8c564b',  # brown
                '#e377c2',  # pink
                '#7f7f7f',  # gray
                '#17becf',  # cyan
                '#bcbd22',  # olive (darker than yellow)
                '#ff9896',  # light red
                '#98df8a',  # light green
                '#c5b0d5',  # light purple
                '#c49c94',  # light brown
                '#f7b6d3',  # light pink
            ]
            
            fig = px.scatter(
                plot_data, 
                x='x', y='y', 
                color='cluster',
                title=title,
                labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
                color_discrete_sequence=readable_colors
            )
        else:
            # Default coloring
            fig = px.scatter(
                plot_data, 
                x='x', y='y',
                title=title,
                labels={'x': 'UMAP 1', 'y': 'UMAP 2'}
            )
        
        # Update traces with custom hover data
        fig.update_traces(
            customdata=custom_data,
            hovertemplate=hover_template,
            marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='white'))
        )
        
        # Update layout for better readability
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial",
                bordercolor="gray"
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def get_cluster_summary_text(self) -> str:
        """
        Generate a text summary of the clustering analysis.
        
        Returns:
            str: Formatted text summary
        """
        if self.cluster_labels is None:
            return "❌ No clustering analysis available. Please run clustering first."
        
        analysis = self.analyze_clusters()
        
        summary = f"📊 **Clustering Analysis Summary**\n\n"
        summary += f"• **Total items:** {analysis['total_items']}\n"
        summary += f"• **Number of clusters:** {analysis['n_clusters']}\n\n"
        
        for cluster in analysis['clusters']:
            summary += f"**Cluster {cluster['id']}** ({cluster['size']} items, {cluster['percentage']:.1f}%)\n"
            
            # Top words
            if cluster['top_words']:
                top_words_str = ", ".join([f"{word} ({count})" for word, count in cluster['top_words'][:5]])
                summary += f"  • Key terms: {top_words_str}\n"
            
            # Sample titles
            if cluster['sample_titles']:
                summary += f"  • Sample titles:\n"
                for title in cluster['sample_titles'][:3]:
                    summary += f"    - {title[:80]}{'...' if len(title) > 80 else ''}\n"
            
            summary += "\n"
        
        return summary


def run_full_clustering_analysis(context: LocalEmbeddingContext, n_clusters: Optional[int] = None) -> Tuple[EmbeddingClusterAnalyzer, str]:
    """
    Run the complete clustering analysis pipeline: k-means clustering + UMAP + analysis.
    
    Args:
        context: LocalEmbeddingContext with built index
        n_clusters: Number of clusters (if None, will find optimal)
        
    Returns:
        Tuple[EmbeddingClusterAnalyzer, str]: Analyzer instance and summary text
    """
    analyzer = EmbeddingClusterAnalyzer(context)
    
    print("🚀 Starting full clustering analysis pipeline...")
    
    # Step 1: Extract embeddings
    if not analyzer.extract_embeddings():
        return analyzer, "❌ Failed to extract embeddings from index."
    
    # Step 2: Perform clustering
    if not analyzer.perform_clustering(n_clusters):
        return analyzer, "❌ Failed to perform clustering."
    
    # Step 3: Perform UMAP reduction
    if not analyzer.perform_umap_reduction():
        return analyzer, "❌ Failed to perform UMAP reduction."
    
    # Step 4: Generate summary
    summary = analyzer.get_cluster_summary_text()
    
    print("✅ Full clustering analysis completed successfully!")
    return analyzer, summary


if __name__ == "__main__":
    # Example usage
    print("This module provides clustering analysis tools for embedding databases.")
    print("Use it with: analyzer = EmbeddingClusterAnalyzer(your_context)")
