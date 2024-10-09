import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import time
load_dotenv()

from filtered_retriever import FilteredRetriever

def timer(seconds):
    print(f"Timer started for {seconds} seconds...")
    time.sleep(seconds)
    print("Time's up!")

def parse_choices(choices_str: str) -> List[str]:
    # First, try splitting by "; "
    choices = choices_str.split("; ")
    
    # If that results in only one choice, try splitting by ","
    if len(choices) == 1:
        choices = choices_str.split(",")
    
    # Remove any leading/trailing whitespace and empty choices
    choices = [choice.strip() for choice in choices if choice.strip()]
    
    # If choices are in "A: text" format, extract just the text
    if all(":" in choice for choice in choices):
        choices = [choice.split(":", 1)[1].strip() for choice in choices]
    
    return choices

def load_arc_challenge_data(data_dir: Path, num_examples: int = 20) -> List[Dict[str, Any]]:
    with open(data_dir / "sources", "r") as f:
        questions = f.read().splitlines()
    
    with open(data_dir / "choices", "r") as f:
        choices = f.read().splitlines()
    
    with open(data_dir / "retrieved_psgs", "r") as f:
        passages = f.read().splitlines()
    
    data = []
    for q, c, p in zip(questions, choices, passages):
        item = {
            "question": q,
            "choices": parse_choices(c),
            "passages": p.split(" [sep] ")
        }
        data.append(item)
    
    # Randomly select num_examples from the dataset
    return random.sample(data, min(num_examples, len(data)))

def create_documents(data: List[Dict[str, Any]]) -> List[Document]:
    documents = []
    for item in data:
        for passage in item["passages"]:
            doc = Document(page_content=passage, metadata={"question": item["question"]})
            documents.append(doc)
    return documents

def setup_retriever(documents: List[Document]):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4")
    
    db = Chroma.from_documents(documents, embeddings)
    base_retriever = db.as_retriever()
    
    filtered_retriever = FilteredRetriever(
        retriever=base_retriever,
        llm=llm,
        embeddings=embeddings
    )
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, filtered_retriever],
        weights=[0.5, 0.5]
    )
    
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

def evaluate_retriever(retriever, data: List[Dict[str, Any]]):
    correct_count = 0
    total_count = len(data)
    
    for item in data:
        question = item["question"]
        choices = item["choices"]
        retrieved_docs = retriever.invoke(question)
        print(retrieved_docs)
        
        timer(6)
        if retrieved_docs in choices:
            correct_count += 1
    
    accuracy = correct_count / total_count
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the ARC Challenge dataset directory")
    parser.add_argument("--num_examples", type=int, default=20, help="Number of examples to use for testing")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Load and preprocess the ARC Challenge data
    arc_data = load_arc_challenge_data(data_dir, args.num_examples)
    documents = create_documents(arc_data)
    
    # Set up the retriever
    retriever = setup_retriever(documents)
    
    # Evaluate the retriever
    accuracy = evaluate_retriever(retriever, arc_data)
    
    print(f"Retriever accuracy on {args.num_examples} ARC Challenge examples: {accuracy:.2%}")

if __name__ == "__main__":
    main()