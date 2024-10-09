
from typing import List, Any, Dict, Union, Optional
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable
from pydantic import Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o")

class FilteredRetriever(RunnableSerializable):
    class Config:
        arbitrary_types_allowed = True

    retriever: Any
    llm: Any
    embeddings: Any
    similarity_threshold: float = Field(default=0.8)

    def remove_redundant_chunks(self, chunks: List[Document]) -> List[Document]:
        texts = [doc.page_content for doc in chunks]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_similarities = cosine_similarity(tfidf_matrix)
        chunks_to_remove = set()

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if i not in chunks_to_remove and j not in chunks_to_remove:
                    if cosine_similarities[i][j] > self.similarity_threshold:
                        if len(chunks[i].page_content) < len(chunks[j].page_content):
                            chunks_to_remove.add(i)
                        else:
                            chunks_to_remove.add(j)

        non_redundant_chunks = [
            chunk for i, chunk in enumerate(chunks) if i not in chunks_to_remove
        ]
        return non_redundant_chunks

    def semantic_similarity(self, query: str, chunks: List[Document]) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)
        chunk_embeddings = self.embeddings.embed_documents(
            [chunk.page_content for chunk in chunks]
        )
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        sorted_chunks = sorted(
            zip(chunks, similarities), key=lambda x: x[1], reverse=True
        )
        return [chunk for chunk, _ in sorted_chunks]

    # def reorder_chunks(self, chunks: List[Document]) -> List[Document]:
    #     n = len(chunks)
    #     reordered = [None] * n
    #     left, right = 0, n - 1
    #     for i, chunk in enumerate(chunks):
    #         if i % 2 == 0:
    #             reordered[left] = chunk
    #             left += 1
    #         else:
    #             reordered[right] = chunk
    #             right -= 1
    #     return reordered

    def invoke(self, input: Union[str, Dict[str, Any]], config: Optional[Dict[str, Any]] = None, **kwargs) -> List[Document]:
        prompt = ChatPromptTemplate.from_template("""You are an AI assistant tasked with determining the relevance of text chunks to user queries. Your job is to analyze the provided chunk and user query, then output either "True" if the chunk is relevant to the query, or "False" if it is not relevant.

              Chunk: {chunk}

              User Query: {query}

              Based on the content of the chunk and the user's query, respond with only "True" or "False" to indicate relevance. Do not provide any explanation or additional text in your response.
          """)
        if isinstance(input, str):
            query = input
        elif isinstance(input, dict):
            query = input.get("query", "")
        else:
            raise ValueError("Input must be either a string or a dictionary with a 'query' key")

        if not query:
            raise ValueError("Query is required")

        chunks = self.retriever.invoke(query)
        chunks = self.remove_redundant_chunks(chunks)
        chunks = self.semantic_similarity(query, chunks)
        # chunks = self.reorder_chunks(chunks)

        filtered_chunks = []
        for chunk in chunks:
            prompt_text = prompt.format(chunk=chunk.page_content, query=query)
            response = self.llm.invoke(prompt_text).content.strip()
            if response.lower() == 'true':
                filtered_chunks.append(chunk)
        return filtered_chunks

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self.invoke(query, **kwargs)