#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Grounding
Augments prompts with retrieved CVE information from local index

Usage:
    python rag_grounding.py \
        --prompts ../../data/prompts/hallu-sec-benchmark.json \
        --index retrieval_index.pkl \
        --model claude-3-5-sonnet-20241022 \
        --output results/rag_results.json \
        --top-k 3
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import re

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def load_index(index_file: Path) -> Dict:
    """Load retrieval index"""
    with open(index_file, 'rb') as f:
        index = pickle.load(f)
    print(f"Loaded {index['type']} index with {len(index['metadata'])} entries")
    return index


def retrieve_documents(index: Dict, query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve top-K relevant documents from index"""
    if index['type'] == 'simple':
        # Keyword-based retrieval
        query_words = query.lower().split()
        scores = {}

        for word in query_words:
            if word in index.get('keyword_index', {}):
                for doc_idx in index['keyword_index'][word]:
                    scores[doc_idx] = scores.get(doc_idx, 0) + 1

        if not scores:
            return []

        top_indices = sorted(scores.keys(), key=lambda x: -scores[x])[:top_k]
        return [index['metadata'][i] for i in top_indices]

    elif 'semantic' in index['type']:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Error: sentence-transformers not available for semantic search")
            return []

        # Semantic retrieval
        model = SentenceTransformer(index['model_name'])
        query_embedding = model.encode([query], convert_to_numpy=True)

        if index['type'] == 'semantic_faiss':
            import faiss
            distances, indices = index['faiss_index'].search(query_embedding.astype('float32'), top_k)
            return [index['metadata'][i] for i in indices[0]]
        else:
            # Numpy-based search
            similarities = np.dot(index['embeddings'], query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [index['metadata'][i] for i in top_indices]

    return []


def format_retrieved_context(documents: List[Dict]) -> str:
    """Format retrieved documents as context"""
    if not documents:
        return "No relevant information found in database."

    context = "Relevant CVE information from authoritative database:\n\n"
    for i, doc in enumerate(documents, 1):
        cve_id = doc.get('cve_id', 'UNKNOWN')
        description = doc.get('description', 'No description')
        severity = doc.get('severity', 'UNKNOWN')
        cvss = doc.get('cvss_score', 'N/A')

        context += f"{i}. {cve_id} (Severity: {severity}, CVSS: {cvss})\n"
        context += f"   {description[:200]}...\n\n"

    return context


def augment_prompt_with_rag(original_prompt: str, retrieved_docs: List[Dict]) -> str:
    """Augment original prompt with retrieved context"""
    context = format_retrieved_context(retrieved_docs)

    augmented = f"""{context}

Based on the authoritative CVE information above, please answer the following question.
Cite specific CVE IDs from the provided context when applicable.
If the information needed is not in the provided context, state that clearly.

Question: {original_prompt}"""

    return augmented


def extract_cve_from_prompt(prompt: str) -> Optional[str]:
    """Extract CVE ID from prompt if present"""
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    matches = re.findall(cve_pattern, prompt, re.IGNORECASE)
    return matches[0].upper() if matches else None


def run_rag_pipeline(
    prompts: List[Dict],
    index: Dict,
    model: str,
    top_k: int = 3
) -> List[Dict]:
    """
    Run RAG pipeline on prompts

    Note: This is a demo implementation. In production, would call actual LLM API
    """
    results = []

    for prompt_data in prompts:
        prompt_id = prompt_data.get('id', '')
        prompt_text = prompt_data.get('prompt', '')

        print(f"\nProcessing: {prompt_id}")

        # Step 1: Retrieve relevant documents
        # First try direct CVE lookup
        cve_id = extract_cve_from_prompt(prompt_text)

        if cve_id:
            # Direct lookup
            print(f"  Direct CVE lookup: {cve_id}")
            retrieved_docs = [
                doc for doc in index['metadata']
                if doc.get('cve_id', '') == cve_id
            ]

            # If not found, fall back to semantic search
            if not retrieved_docs:
                print(f"  CVE not found, using semantic search...")
                retrieved_docs = retrieve_documents(index, prompt_text, top_k)
        else:
            # Semantic search
            print(f"  Semantic search...")
            retrieved_docs = retrieve_documents(index, prompt_text, top_k)

        print(f"  Retrieved {len(retrieved_docs)} documents")

        # Step 2: Augment prompt
        augmented_prompt = augment_prompt_with_rag(prompt_text, retrieved_docs)

        # Step 3: Query LLM (demo - would call actual API)
        llm_response = f"[DEMO MODE] Would query {model} with augmented prompt containing {len(retrieved_docs)} retrieved documents"

        # Store result
        results.append({
            'prompt_id': prompt_id,
            'original_prompt': prompt_text,
            'retrieved_documents': [doc.get('cve_id', 'UNKNOWN') for doc in retrieved_docs],
            'augmented_prompt': augmented_prompt,
            'model': model,
            'response': llm_response,
            'rag_enabled': True
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="RAG grounding for hallucination mitigation")
    parser.add_argument('--prompts', type=str, required=True, help='Input prompts JSON file')
    parser.add_argument('--index', type=str, required=True, help='Retrieval index pickle file')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20241022', help='Model to use')
    parser.add_argument('--top-k', type=int, default=3, help='Number of documents to retrieve')
    parser.add_argument('--output', type=str, required=True, help='Output results file')
    parser.add_argument('--max-prompts', type=int, default=None, help='Max prompts to process (for testing)')

    args = parser.parse_args()

    # Load index
    print("Loading retrieval index...")
    index = load_index(Path(args.index))

    # Load prompts
    print("\nLoading prompts...")
    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)

    # Handle different formats
    if isinstance(prompts_data, dict) and 'prompts' in prompts_data:
        prompts = prompts_data['prompts']
    elif isinstance(prompts_data, list):
        prompts = prompts_data
    else:
        prompts = []

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    print(f"Loaded {len(prompts)} prompts")

    # Run RAG pipeline
    print("\nRunning RAG pipeline...")
    print("="*60)
    results = run_rag_pipeline(prompts, index, args.model, args.top_k)

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'model': args.model,
        'rag_enabled': True,
        'top_k': args.top_k,
        'index_type': index['type'],
        'total_prompts': len(results),
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print(f"  Processed: {len(results)} prompts")
    print(f"  RAG-augmented responses generated")

    print("\nNote: This is a DEMO implementation.")
    print("For production use:")
    print("  1. Implement actual LLM API calls")
    print("  2. Add response parsing and verification")
    print("  3. Integrate with baseline for comparison")


if __name__ == '__main__':
    main()
