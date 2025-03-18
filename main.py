import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage


# Set API Key (Store securely in production)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize Graph
graph_builder = StateGraph(MessagesState)


def init_llm():
    """Initialize the language model."""
    return init_chat_model("llama3-8b-8192", model_provider="groq")


def init_vector_store():
    """Initialize the in-memory vector store with embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return InMemoryVectorStore(embeddings)


def process_url(url, vector_store):
    """Load content from a URL, split into chunks, and store in vector store."""
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(documents=all_splits)
    return all_splits


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve relevant information based on the query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)

    if not retrieved_docs:
        return "No relevant data found.", []

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def generate(state: MessagesState):
    """Generate a response using retrieved content."""
    recent_tool_messages = [
        msg for msg in reversed(state["messages"]) if msg.type == "tool"
    ]
    docs_content = "\n\n".join(doc.content for doc in recent_tool_messages[::-1])

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If the answer is unknown, state it clearly.\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        msg
        for msg in state["messages"]
        if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
    ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}


def build_graph():
    """Construct the AI workflow graph."""
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(ToolNode([retrieve]))
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond", tools_condition, {END: END, "tools": "tools"}
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()


def main():
    st.title("Web Content Q&A Tool")

    global llm, vector_store
    llm = init_llm()
    vector_store = init_vector_store()
    graph = build_graph()

    input_url = st.text_input("Enter URL to process:")

    if input_url:
        st.info("Fetching and processing content...")
        docs = process_url(input_url, vector_store)

        if docs:
            st.success("Content successfully ingested! You can now ask questions.")
        else:
            st.error("Failed to retrieve content from the provided URL.")

    input_query = st.text_input("Enter your question:")

    if input_query and input_url:
        with st.spinner("Fetching answer..."):
            for step in graph.stream(
                {"messages": [{"role": "user", "content": input_query}]},
                stream_mode="values",
            ):
                st.write(step["messages"][-1].content)
    elif input_query and not input_url:
        st.warning("Please enter a URL first.")


if __name__ == "__main__":
    main()
