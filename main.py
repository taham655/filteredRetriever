from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from filtered_retriever import FilteredRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from typing import List, Any

from langchain_chroma import Chroma



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o")

loader = PyPDFLoader("build-career-in-ai.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
documents = text_splitter.split_documents(pages)

print("here 1")

db = Chroma.from_documents(documents, OpenAIEmbeddings())


retriever = db.as_retriever()

filtered_retriever = FilteredRetriever(
    retriever=retriever,
    llm=llm,
    embeddings=embeddings
)

bm25_retriever = BM25Retriever.from_documents(documents)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, filtered_retriever], weights=[0.5, 0.5]
)


# print(filtered_retriever.invoke("How excel in AI?"))


#reranking

compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

print(compression_retriever.invoke("How excel in AI?"))

