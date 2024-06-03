import os
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings

if TYPE_CHECKING:
    from langchain_community.document_loaders.pdf import PyPDFLoader
    from langchain_community.retrievers.wikipedia import WikipediaRetriever
    from langchain_community.vectorstores.chroma import Chroma
else:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.retrievers import WikipediaRetriever
    from langchain_community.vectorstores import Chroma

from backend.tools.retrieval.base import BaseRetrieval

"""
Plug in your lang chain retrieval implementation here. 
We have an example flows with wikipedia and vector DBs.

More details: https://python.langchain.com/docs/integrations/retrievers
"""


class LangChainWikiRetriever(BaseRetrieval):
    """
    This class retrieves documents from Wikipedia using the langchain package.
    This requires wikipedia package to be installed.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    def is_available(cls) -> bool:
        return True

    def retrieve_documents(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        wiki_retriever = WikipediaRetriever()
        docs = wiki_retriever.get_relevant_documents(query)
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        documents = text_splitter.split_documents(docs)
        return [
            {
                "text": doc.page_content,
                "title": doc.metadata.get("title", None),
                "url": doc.metadata.get("source", None),
            }
            for doc in documents
        ]


class LangChainVectorDBRetriever(BaseRetrieval):
    """
    This class retrieves documents from a vector database using the langchain package.
    """

    cohere_api_key = os.environ.get("COHERE_API_KEY")

    def __init__(self, filepath: str):
        path = Path(filepath)
        self.filepath = str(path)
        self.persist_directory = str(path.parent)
        self.file_id = path.name

    @classmethod
    def is_available(cls) -> bool:
        return cls.cohere_api_key is not None

    def retrieve_documents(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        cohere_embeddings = CohereEmbeddings(cohere_api_key=self.cohere_api_key)
        # Load text files and split into chunks
        loader = PyPDFLoader(self.filepath)
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        pages = loader.load_and_split(text_splitter)
        # Create a vector store from the documents
        db = Chroma.from_documents(
            documents=pages,
            ids=[f"{self.file_id}_{i}" for i in range(len(pages))],
            embedding=cohere_embeddings,
            persist_directory=self.persist_directory,
        )
        input_docs = db.as_retriever(
            search_kwargs={"filter": {"source": self.filepath}}
        ).get_relevant_documents(query)
        return [dict({"text": doc.page_content, "title": self.file_id}) for doc in input_docs]
