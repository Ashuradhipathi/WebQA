from langgraph.graph import MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

# Globals (to be set in main.py)
llm = None
vector_store = None

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve relevant information from the vector store."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    if not retrieved_docs:
        return "No relevant data found.", []
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" 
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """
    Generate a tool call for retrieval.
    Uses the LLM bound with our retrieval tool.
    """
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate(state: MessagesState):
    """
    Generate a final answer using only the retrieved context.
    The prompt explicitly instructs the LLM to rely solely on the provided context.
    """
    # Gather retrieved context from tool messages.
    recent_tool_messages = [msg for msg in reversed(state["messages"]) if msg.type == "tool"]
    retrieved_context = "\n\n".join(msg.content for msg in recent_tool_messages[::-1])
    
    # Strong instruction: use only the retrieved context.
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Keep the answers precise and concise"
        "Only use the following retrieved context to answer the question. "
        "Do not use any external or pre-trained knowledge. "
        "If the answer is not found in the context, state that you do not know.\n\n"
        f"{retrieved_context}"
    )
    
    conversation_messages = [
        msg for msg in state["messages"]
        if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
    ]
    
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

def build_graph(graph_builder):
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
