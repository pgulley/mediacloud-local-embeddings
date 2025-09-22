#!/usr/bin/env python3
"""
Demo script showing how to use the new clustering analysis functionality.

This script demonstrates:
1. Building an embedding index from MediaCloud
2. Running k-means clustering and UMAP dimensionality reduction
3. Generating cluster analysis and visualization

Usage:
    python demo_clustering.py
"""

import os
from EmbeddingContext import LocalEmbeddingContext
from clustering import EmbeddingClusterAnalyzer, run_full_clustering_analysis

def main():
    print("🚀 Clustering Analysis Demo")
    print("=" * 50)
    
    # Check if we have required environment variables
    if not os.getenv("MC_API_KEY"):
        print("❌ Please set MC_API_KEY environment variable")
        print("   You can create a .env file with: MC_API_KEY=your_api_key_here")
        return
    
    print("\n📚 Step 1: Building embedding index...")
    print("This will fetch stories from MediaCloud and build an embedding index.")
    
    # Create context and build index with a simple query
    context = LocalEmbeddingContext()
    
    # Use a small query for demo purposes
    query = "climate change"
    days = 7  # Last 7 days
    
    print(f"Query: {query}")
    print(f"Days: {days}")
    
    try:
        context.build_index_from_query(query, days)
        count = context.count()
        print(f"✅ Index built with {count} chunks")
        
        if count == 0:
            print("❌ No content found. Try a different query or increase days.")
            return
            
    except Exception as e:
        print(f"❌ Error building index: {e}")
        print("Make sure your MediaCloud API key is valid and you have internet access.")
        return
    
    print("\n🔬 Step 2: Running clustering analysis...")
    print("This will perform k-means clustering and UMAP dimensionality reduction.")
    
    try:
        # Run the full clustering analysis pipeline
        analyzer, summary = run_full_clustering_analysis(context, n_clusters=None)  # Auto-detect clusters
        
        print(f"✅ Analysis completed!")
        print(f"   - Found {analyzer.n_clusters} clusters")
        print(f"   - Processed {len(analyzer.embeddings)} embeddings")
        print(f"   - Reduced to 2D using UMAP")
        
        except Exception as e:
        print(f"❌ Error in clustering analysis: {e}")
        return
    
    print("\n📊 Step 3: Analysis Results")
    print("=" * 30)
    print(summary)
    
    print("\n🎨 Step 4: Generating visualization...")
    try:
        fig = analyzer.create_scatter_plot(
            color_by='cluster',
            title=f"UMAP Visualization: {query}"
        )
        
        # Save the plot as HTML
        output_file = "cluster_visualization.html"
        fig.write_html(output_file)
        print(f"✅ Visualization saved as {output_file}")
        print(f"   Open this file in your browser to view the interactive plot.")
        
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
    
    print("\n🎉 Demo completed!")
    print("\nTo use this in the Gradio app:")
    print("1. Run: python gradio_app.py")
    print("2. Build an index in the 'Build Index' tab")
    print("3. Go to the 'Clustering' tab")
    print("4. Click 'Run Clustering Analysis' to see clustering and visualization")

if __name__ == "__main__":
    main()
