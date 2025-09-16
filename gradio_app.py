import gradio as gr
import pandas as pd
import time
from typing import Optional, Tuple, List

from EmbeddingContext import LocalEmbeddingContext
from summarize import (
    LLMSettings,
    get_llm_from_settings,
    summarize_single_pass,
    summarize_map_reduce,
)

# Global state
context: Optional[LocalEmbeddingContext] = None
last_search_results: Optional[pd.DataFrame] = None
last_query: str = ""
# Global settings
global_model: str = "llama3:8b"
global_temperature: float = 0.2


def build_index(query: str, days: int, progress=gr.Progress()) -> Tuple[str, str, str]:
    """Build index from MediaCloud query with progress updates."""
    global context
    
    if not query.strip():
        return "❌ Please provide a MediaCloud query.", "", "disabled"
    
    progress(0, desc="Initializing...")
    
    try:
        # Initialize context
        if context is None:
            context = LocalEmbeddingContext()
        
        # Progress callback
        def progress_callback(p):
            stories = p.get("stories", 0)
            chunks = p.get("chunks", 0)
            kept = p.get("kept", 0)
            progress(0.5, desc=f"Processing: {stories} stories, {chunks} chunks, {kept} kept")
        
        # Build index
        context.build_index_from_query(query, days, progress_callback=progress_callback)
        count = context.count()
        
        progress(1.0, desc="Complete!")
        
        return (
            f"✅ Index built successfully! Kept {count} chunks from the last {days} days.",
            f"Query: {query}\nDays: {days}\nChunks: {count}",
            "enabled"
        )
        
    except Exception as e:
        return f"❌ Error building index: {str(e)}", "", "disabled"


def search_index(query: str, k: int = 20) -> Tuple[str, pd.DataFrame, str]:
    """Search the built index."""
    global context, last_search_results, last_query
    
    if context is None or context.index is None:
        return "❌ Please build an index first.", pd.DataFrame(), "disabled"
    
    if not query.strip():
        return "❌ Please provide a search query.", pd.DataFrame(), "disabled"
    
    try:
        results = context.search(query, k=k)
        last_search_results = results
        last_query = query
        
        if results.empty:
            return "🔍 No results found for your query.", pd.DataFrame(), "disabled"
        
        # Format results for display
        display_results = results[['score', 'title', 'publish_date', 'text']].copy()
        display_results['text'] = display_results['text'].str[:200] + "..."
        display_results['score'] = display_results['score'].round(3)
        
        message = f"🔍 Found {len(results)} results for: '{query}'"
        
        return message, display_results, "enabled"
        
    except Exception as e:
        return f"❌ Search error: {str(e)}", pd.DataFrame(), "disabled"


def search_and_summarize(
    query: str,
    k: int,
    top_n: int, 
    mode: str, 
    style: str,
    progress=gr.Progress()
):
    """Search index and generate summary with streaming."""
    global context, global_model, global_temperature
    
    if context is None or context.index is None:
        yield "❌ Please build an index first.", ""
        return
    
    if not query.strip():
        yield "❌ Please provide a search query.", ""
        return
    
    try:
        progress(0.1, desc="Searching index...")
        
        # Search first
        results = context.search(query, k=k)
        
        if results.empty:
            yield "🔍 No results found for your query.", ""
            return
        
        progress(0.3, desc="Preparing documents for summary...")
        
        # Prepare documents for summarization
        docs = results.head(top_n).to_dict(orient="records")
        
        # Set up LLM settings
        settings = LLMSettings(
            llm_model=global_model,
            llm_temperature=global_temperature,
        )
        
        search_status = f"🔍 Found {len(results)} results for: '{query}' → Generating summary..."
        
        progress(0.5, desc="Generating summary...")
        
        # Generate summary with streaming
        if mode == "single-pass":
            stream = summarize_single_pass(
                docs, 
                query=query, 
                style=style, 
                settings=settings, 
                stream=True
            )
        else:
            stream = summarize_map_reduce(
                docs, 
                query=query, 
                settings=settings, 
                stream=True
            )
        
        # Stream the tokens as they arrive
        summary_text = "🤖 **Summary**\n\n"
        for token in stream:
            summary_text += token
            yield search_status, summary_text
        
        progress(0.8, desc="Adding citations...")
        
        # Add citations
        citation_bundle = context.build_citation_bundle(results.head(top_n))
        
        if citation_bundle["citations"]:
            citations_text = "\n\n📚 **Sources:**\n"
            for cite in citation_bundle["citations"]:
                citations_text += f"{cite['id']} {cite['title']} ({cite['publish_date']})\n"
            summary_text += citations_text
        
        progress(1.0, desc="Complete!")
        
        final_status = f"🔍 Found {len(results)} results → Summarized top {top_n} ✅"
        yield final_status, summary_text
        
    except Exception as e:
        yield f"❌ Error: {str(e)}", ""


def generate_summary(
    top_n: int, 
    mode: str, 
    style: str,
    progress=gr.Progress()
):
    """Generate summary from existing search results with streaming."""
    global context, last_search_results, last_query, global_model, global_temperature
    
    if last_search_results is None or last_search_results.empty:
        yield "❌ No search results to summarize. Please search first."
        return
    
    try:
        progress(0, desc="Preparing documents...")
        
        # Prepare documents
        docs = last_search_results.head(top_n).to_dict(orient="records")
        
        # Set up LLM settings
        settings = LLMSettings(
            llm_model=global_model,
            llm_temperature=global_temperature,
        )
        
        progress(0.3, desc="Generating summary...")
        
        # Generate summary with streaming
        if mode == "single-pass":
            stream = summarize_single_pass(
                docs, 
                query=last_query, 
                style=style, 
                settings=settings, 
                stream=True
            )
        else:
            stream = summarize_map_reduce(
                docs, 
                query=last_query, 
                settings=settings, 
                stream=True
            )
        
        # Stream the tokens as they arrive
        summary_text = "🤖 **Summary**\n\n"
        for token in stream:
            summary_text += token
            yield summary_text
        
        progress(0.8, desc="Adding citations...")
        
        # Add citations
        citation_bundle = context.build_citation_bundle(last_search_results.head(top_n))
        
        if citation_bundle["citations"]:
            citations_text = "\n\n📚 **Sources:**\n"
            for cite in citation_bundle["citations"]:
                citations_text += f"{cite['id']} {cite['title']} ({cite['publish_date']})\n"
            summary_text += citations_text
        
        progress(1.0, desc="Complete!")
        yield summary_text
        
    except Exception as e:
        yield f"❌ Summarization error: {str(e)}"


def ask_question(question: str, k: int = 10) -> Tuple[str, pd.DataFrame]:
    """Ask a question about the indexed content."""
    global context
    
    if context is None or context.index is None:
        return "❌ Please build an index first.", pd.DataFrame()
    
    if not question.strip():
        return "❌ Please provide a question.", pd.DataFrame()
    
    try:
        results = context.qa(question, k=k)
        
        if results.empty:
            return "🤔 No relevant content found for your question.", pd.DataFrame()
        
        # Format results for display
        display_results = results[['score', 'title', 'publish_date', 'text']].copy()
        display_results['text'] = display_results['text'].str[:300] + "..."
        display_results['score'] = display_results['score'].round(3)
        
        message = f"❓ **Question:** {question}\n\n📋 Found {len(results)} relevant pieces of content:"
        
        return message, display_results
        
    except Exception as e:
        return f"❌ Q&A error: {str(e)}", pd.DataFrame()


def get_attention_chart() -> str:
    """Get attention over time data."""
    global context
    
    if context is None:
        return "❌ Please build an index first."
    
    try:
        aot_df = context.attention_over_time()
        
        if aot_df.empty:
            return "📈 No attention data available."
        
        # Format as text table
        content = "📈 **Attention Over Time**\n\n```\n"
        content += f"{'Date':<12} {'Original':<10} {'Semantic':<10}\n"
        content += "-" * 35 + "\n"
        
        for _, row in aot_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'Unknown'
            orig = int(row['original'])
            sem = int(row['semantic'])
            content += f"{date_str:<12} {orig:<10} {sem:<10}\n"
        
        content += "```"
        return content
        
    except Exception as e:
        return f"❌ Attention chart error: {str(e)}"


def update_settings(model: str, temperature: float) -> str:
    """Update global LLM settings."""
    global global_model, global_temperature
    
    global_model = model
    global_temperature = temperature
    
    return f"✅ Settings updated: Model={model}, Temperature={temperature}"


def reset_context() -> Tuple[str, str, str, str, pd.DataFrame, pd.DataFrame]:
    """Reset the application state."""
    global context, last_search_results, last_query
    
    context = None
    last_search_results = None
    last_query = ""
    
    return (
        "🔄 Application reset. Ready to build a new index!",
        "",  # index_info
        "",  # search_output
        "",  # summary_output
        pd.DataFrame(),  # search_results
        pd.DataFrame(),  # qa_results
    )


# Create Gradio interface
with gr.Blocks(title="Featherweight Embeddings: MediaCloud Semantic Search", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🚀 Featherweight Embeddings: MediaCloud Semantic Search")
    gr.Markdown("Build semantic search indexes from MediaCloud stories and generate AI summaries.")
    
    with gr.Tab("📚 Build Index"):
        gr.Markdown("## Step 1: Build Index from MediaCloud")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="MediaCloud Query",
                    placeholder="(police OR victim*) AND \"gun control\"",
                    value="(police OR victim* OR crim* OR prison OR arrest* OR suspect) AND (\"gun control\" OR \"gun restrict\"~5)",
                    lines=3,
                    info="Boolean operators supported: AND, OR, NOT, quotes for phrases"
                )
            with gr.Column(scale=1):
                days_input = gr.Slider(
                    label="Days Lookback",
                    minimum=1,
                    maximum=180,
                    value=30,
                    step=1,
                    info="Days to look back from today"
                )
        
        build_btn = gr.Button("🔨 Build Index", variant="primary", size="lg")
        build_output = gr.Textbox(label="Status", interactive=False)
        index_info = gr.Textbox(label="Index Info", interactive=False)
        
        build_btn.click(
            build_index,
            inputs=[query_input, days_input],
            outputs=[build_output, index_info, build_btn]
        )
    
    with gr.Tab("🔍 Search"):
        gr.Markdown("## Search the Index")
        
        with gr.Row():
            search_input = gr.Textbox(
                label="Search Query",
                placeholder="stories about minnesota",
                scale=3
            )
            search_k = gr.Slider(
                label="Top K Results",
                minimum=5,
                maximum=50,
                value=20,
                step=1,
                scale=1
            )
        
        search_btn = gr.Button("🔍 Search", variant="primary")
        search_output = gr.Textbox(label="Search Status", interactive=False)
        search_results = gr.Dataframe(
            label="Search Results",
            headers=["Score", "Title", "Date", "Text Preview"],
            interactive=False
        )
        
        gr.Markdown("## Generate Summary from Results")
        
        with gr.Row():
            with gr.Column():
                top_n = gr.Slider(label="Summarize Top N", minimum=3, maximum=20, value=10, step=1)
                mode = gr.Radio(["single-pass", "map-reduce"], label="Mode", value="single-pass")
                style = gr.Radio(["bullets", "paragraph"], label="Style", value="bullets")
        
        summarize_btn = gr.Button("📝 Generate Summary", variant="secondary")
        summary_output = gr.Textbox(label="Summary", interactive=False, lines=10)
        
        search_btn.click(
            search_index,
            inputs=[search_input, search_k],
            outputs=[search_output, search_results, summarize_btn]
        )
        
        summarize_btn.click(
            generate_summary,
            inputs=[top_n, mode, style],
            outputs=[summary_output]
        )
    
    with gr.Tab("📝 Summarize"):
        gr.Markdown("## Search & Summarize in One Step")
        gr.Markdown("Enter a query to search the index and generate an AI summary with citations.")
        
        with gr.Row():
            with gr.Column(scale=2):
                summary_query = gr.Textbox(
                    label="Search Query",
                    placeholder="stories about minnesota gun laws",
                    info="What topic would you like summarized?"
                )
            with gr.Column(scale=1):
                summary_search_k = gr.Slider(
                    label="Search Results",
                    minimum=10,
                    maximum=100,
                    value=30,
                    step=5,
                    info="Number of results to search"
                )
        
        with gr.Row():
            with gr.Column():
                summary_top_n = gr.Slider(
                    label="Summarize Top N", 
                    minimum=3, 
                    maximum=20, 
                    value=10, 
                    step=1,
                    info="How many top results to include in summary"
                )
                summary_mode = gr.Radio(
                    ["single-pass", "map-reduce"], 
                    label="Summary Mode", 
                    value="single-pass",
                    info="single-pass: faster, map-reduce: better for many docs"
                )
            with gr.Column():
                summary_style = gr.Radio(
                    ["bullets", "paragraph"], 
                    label="Summary Style", 
                    value="bullets",
                    info="Format of the generated summary"
                )
        
        summarize_search_btn = gr.Button("🔍📝 Search & Summarize", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                summarize_search_status = gr.Textbox(label="Search Status", interactive=False)
            with gr.Column():
                summarize_output_main = gr.Textbox(label="AI Summary with Citations", interactive=False, lines=15)
        
        summarize_search_btn.click(
            search_and_summarize,
            inputs=[summary_query, summary_search_k, summary_top_n, summary_mode, summary_style],
            outputs=[summarize_search_status, summarize_output_main]
        )
    
    with gr.Tab("❓ Ask Questions"):
        gr.Markdown("## Ask Questions About the Index")
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Question",
                placeholder="What are the main themes in these stories?",
                scale=3
            )
            qa_k = gr.Slider(
                label="Top K Results",
                minimum=5,
                maximum=20,
                value=10,
                step=1,
                scale=1
            )
        
        qa_btn = gr.Button("❓ Ask Question", variant="primary")
        qa_output = gr.Textbox(label="Answer", interactive=False, lines=5)
        qa_results = gr.Dataframe(
            label="Relevant Content",
            headers=["Score", "Title", "Date", "Text Preview"],
            interactive=False
        )
        
        qa_btn.click(
            ask_question,
            inputs=[question_input, qa_k],
            outputs=[qa_output, qa_results]
        )
    
    with gr.Tab("📈 Analytics"):
        gr.Markdown("## Attention Over Time")
        
        chart_btn = gr.Button("📈 Show Attention Chart", variant="secondary")
        chart_output = gr.Textbox(label="Attention Data", interactive=False, lines=15)
        
        chart_btn.click(
            get_attention_chart,
            outputs=[chart_output]
        )
    
    with gr.Tab("⚙️ Settings"):
        gr.Markdown("## LLM Settings")
        
        with gr.Row():
            with gr.Column():
                model_input = gr.Textbox(label="Model", value="llama3:8b")
                temperature_input = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.2, step=0.05)
            with gr.Column():
                settings_btn = gr.Button("💾 Update Settings", variant="secondary")
                settings_output = gr.Textbox(label="Settings Status", interactive=False)
        
        settings_btn.click(
            update_settings,
            inputs=[model_input, temperature_input],
            outputs=[settings_output]
        )
        
        gr.Markdown("## Application Management")
        
        reset_btn = gr.Button("🔄 Reset Application", variant="stop")
        reset_output = gr.Textbox(label="Reset Status", interactive=False)
        
        reset_btn.click(
            reset_context,
            outputs=[reset_output, index_info, search_output, summary_output, search_results, qa_results]
        )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
