from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_url(url, vector_store):
    """
    Load content from a URL, split it into manageable chunks,
    and add them to the provided vector store.
    """
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    return all_splits
