import os
import asyncio
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import MessagesState, StateGraph
from llm_utils import init_llm, init_vector_store
from ingestion import process_url
from graph_builder import build_graph

# Ensure an asyncio event loop exists (avoids "no running event loop" errors).
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Ensure USER_AGENT is set.
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "WebQA/1.0"  # Adjust if needed.

# Create a global ThreadPoolExecutor for background tasks.
executor = ThreadPoolExecutor(max_workers=2)

def main():
    st.title("Web Content Q&A Tool")
    
    # Initialize LLM and vector store.
    llm = init_llm()
    vector_store = init_vector_store()
    
    # Set the globals in graph_builder so that our nodes can access them.
    import graph_builder
    graph_builder.llm = llm
    graph_builder.vector_store = vector_store

    input_url = st.text_input("Enter URL to process:")
    
    # Use session state to cache ingestion and graph building.
    if "ingested" not in st.session_state:
        st.session_state.ingested = False
    if "graph" not in st.session_state:
        st.session_state.graph = None

    if input_url and not st.session_state.ingested:
        st.info("Fetching and processing content in the background...")
        # Run ingestion in a background thread.
        future = executor.submit(process_url, input_url, vector_store)
        try:
            docs = future.result(timeout=30)  # wait up to 30 seconds
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            docs = None

        if docs:
            st.session_state.ingested = True
            st.success("Content successfully ingested! You can now ask questions.")
            # Build the workflow graph with the required state schema.
            graph_builder_instance = StateGraph(MessagesState)
            st.session_state.graph = build_graph(graph_builder_instance)
        else:
            st.error("Failed to retrieve content from the provided URL.")

    input_query = st.text_input("Enter your question:")

    if input_query and st.session_state.ingested and st.session_state.graph:
        with st.spinner("Fetching answer..."):
            # Stream the answer step by step.
            for step in st.session_state.graph.stream(
                {"messages": [{"role": "user", "content": input_query}]},
                stream_mode="values",
            ):
                st.write(step["messages"][-1].content)
    elif input_query and not st.session_state.ingested:
        st.warning("Please enter a URL and wait for ingestion to complete.")

if __name__ == "__main__":
    main()
