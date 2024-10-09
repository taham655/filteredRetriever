import os
import argparse
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from collections import Counter
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from filtered_retriever import FilteredRetriever
import random
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

def timer(seconds):
    print(f"Timer started for {seconds} seconds...")
    time.sleep(seconds)
    print("Time's up!")

# Start a 10-second timer


def load_popqa_data(data_dir: str, num_examples: int = 100) -> List[Dict[str, Any]]:
    file_path = os.path.join(data_dir, 'test_popqa.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'test_popqa.txt' file not found in {data_dir}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    all_passages = []
    current_question = None

    for line in lines:
        line = line.strip()
        if '[SEP]' in line:
            question, passage = line.split('[SEP]')
            question = question.strip()
            passage = passage.strip()
            if passage:
                all_passages.append({"question": question, "text": passage})

    # Randomly select num_examples
    selected_passages = random.sample(all_passages, min(num_examples, len(all_passages)))

    # Group passages by question
    grouped_data = {}
    for passage in selected_passages:
        if passage["question"] not in grouped_data:
            grouped_data[passage["question"]] = []
        grouped_data[passage["question"]].append(passage["text"])

    dataset = [{"question": q, "ctxs": [{"text": t} for t in texts]} for q, texts in grouped_data.items()]

    print(f"Loaded {len(dataset)} items from the dataset.")
    return dataset

def create_documents_from_popqa(dataset: List[Dict[str, Any]], num_documents: int = 1000) -> List[Document]:
    documents = []
    for item in dataset:
        question = item["question"]
        contexts = item["ctxs"]
        
        # Randomly select contexts if there are more than num_documents / len(dataset)
        num_docs_per_question = num_documents // len(dataset)
        if len(contexts) > num_docs_per_question:
            selected_contexts = random.sample(contexts, num_docs_per_question)
        else:
            selected_contexts = contexts
        
        for ctx in selected_contexts:
            documents.append(Document(
                page_content=ctx["text"],
                metadata={"question": question}
            ))
    
    print(f"Created {len(documents)} documents from {len(dataset)} dataset items.")
    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def create_chroma_db(documents: List[Document], embeddings, batch_size: int = 500) -> Chroma:
    db = Chroma.from_documents(documents[:1], embeddings)  # Initialize with one document
    for i in tqdm(range(1, len(documents), batch_size), desc="Creating Chroma DB"):
        batch = documents[i:i+batch_size]
        db.add_documents(batch)
    return db

def evaluate_retriever(retriever, dataset: List[Dict[str, Any]], top_k: int = 5, time_out: int = None) -> Dict[str, float]:
    metrics = Counter()
    total = len(dataset)
    questions_with_retrievals = 0

    for item in tqdm(dataset, desc="Evaluating", unit="question"):
        question = item["question"]
        ground_truth_passages = set(ctx["text"] for ctx in item["ctxs"])

        timer(time_out)
        retrieved_docs = retriever.invoke(question)[:top_k]
        retrieved_passages = [doc.page_content for doc in retrieved_docs]
        
        if not retrieved_passages:
            continue  # Skip this question if no passages were retrieved
        
        questions_with_retrievals += 1
        
        # Exact match
        if any(gt == rp for gt in ground_truth_passages for rp in retrieved_passages):
            print("GroundTruth:", )
            metrics['exact_match'] += 1
        
        # Partial match
        if any(gt in rp or rp in gt for gt in ground_truth_passages for rp in retrieved_passages):
            metrics['partial_match'] += 1
        
        # Relevance score
        relevant_passages = sum(1 for rp in retrieved_passages if any(gt in rp or rp in gt for gt in ground_truth_passages))
        metrics['relevance_score'] += relevant_passages / len(retrieved_passages)

    # Avoid division by zero
    if questions_with_retrievals == 0:
        return {
            "exact_match_accuracy": 0,
            "partial_match_accuracy": 0,
            "average_relevance_score": 0,
            "retrieval_rate": 0
        }

    results = {
        "exact_match_accuracy": metrics['exact_match'] / questions_with_retrievals,
        "partial_match_accuracy": metrics['partial_match'] / questions_with_retrievals,
        "average_relevance_score": metrics['relevance_score'] / questions_with_retrievals,
        "retrieval_rate": questions_with_retrievals / total
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Test retriever on PopQA dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to PopQA dataset directory")
    parser.add_argument("--db_dir", type=str, default="./chroma_db", help="Path to persist Chroma database")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to process")
    parser.add_argument("--num_documents", type=int, default=1000, help="Total number of documents to process")
    args = parser.parse_args()

    # Load data
    try:
        popqa_data = load_popqa_data(args.data_dir, num_examples=args.num_examples)
        print(f"Successfully loaded {len(popqa_data)} items from the dataset.")
        
        # Print a sample item for verification
        if popqa_data:
            print("\nSample item:")
            print(f"Question: {popqa_data[0]['question']}")
            print(f"Number of passages: {len(popqa_data[0]['ctxs'])}")
            print(f"First passage: {popqa_data[0]['ctxs'][0]['text'][:100]}...")
    except Exception as e:
        print(f"Error loading PopQA data: {str(e)}")
        return

    # Create documents from PopQA data
    documents = create_documents_from_popqa(popqa_data, num_documents=args.num_documents)

    # Chunk documents
    print("Chunking documents...")
    chunked_documents = chunk_documents(documents)
    print(f"Created {len(chunked_documents)} chunks from {len(documents)} original documents.")

    # Initialize components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create Chroma database
    print("Creating Chroma database...")
    db = create_chroma_db(chunked_documents, embeddings)
    base_retriever = db.as_retriever()

    # Create retrievers
    print("Initializing retrievers...")
    filtered_retriever = FilteredRetriever(
        retriever=base_retriever,
        llm=llm,
        embeddings=embeddings
    )

    bm25_retriever = BM25Retriever.from_documents(chunked_documents)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, filtered_retriever], weights=[0.5, 0.5]
    )

    # Reranking
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    # Evaluate retrievers
    for retriever_name, retriever in [
        # ("Base Retriever", base_retriever),
        # ("Filtered Retriever", filtered_retriever),
        # ("BM25 Retriever", bm25_retriever),
        # ("Ensemble Retriever", ensemble_retriever),
        ("Compression Retriever", compression_retriever)
    ]:
        if retriever_name == "Compression Retriever":
            time_out = 6
        else:
            time_out = None
        print(f"\nEvaluating {retriever_name}:")
        results = evaluate_retriever(retriever, popqa_data, top_k=5, time_out=time_out)
        print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
        print(f"Partial Match Accuracy: {results['partial_match_accuracy']:.4f}")
        print(f"Average Relevance Score: {results['average_relevance_score']:.4f}")
        print(f"Retrieval Rate: {results['retrieval_rate']:.4f}")

if __name__ == "__main__":
    main()