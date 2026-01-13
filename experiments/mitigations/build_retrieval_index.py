#!/usr/bin/env python3
"""
Build Retrieval Index for RAG
Creates a searchable index from NVD CVE metadata for RAG grounding

Usage:
    python build_retrieval_index.py \
        --nvd-data ../../data/gold/nvd_metadata.json \
        --output retrieval_index.pkl
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np

# Optional: use sentence transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

# Optional: use FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Install with: pip install faiss-cpu")


def load_nvd_metadata(nvd_file: Path) -> List[Dict]:
    """Load NVD CVE metadata"""
    if not nvd_file.exists():
        raise FileNotFoundError(
            f"NVD metadata file not found: {nvd_file}\n"
            f"Please provide a valid NVD data file. You can fetch NVD data using:\n"
            f"  python data/scripts/fetch_nvd_metadata.py --output {nvd_file}"
        )

    with open(nvd_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'cve_items' in data:
        return data['cve_items']

    return []


def build_text_corpus(nvd_data: List[Dict]) -> List[str]:
    """Build text corpus from NVD metadata"""
    corpus = []
    for item in nvd_data:
        # Combine CVE ID and description for retrieval
        text = f"{item['cve_id']}: {item.get('description', '')}"
        corpus.append(text)
    return corpus


def build_simple_index(nvd_data: List[Dict]) -> Dict:
    """Build simple keyword-based index (fallback if no sentence transformers)"""
    index = {
        'cve_ids': {},
        'keyword_index': {},
        'metadata': nvd_data
    }

    # Build CVE ID lookup
    for i, item in enumerate(nvd_data):
        cve_id = item.get('cve_id', '')
        index['cve_ids'][cve_id] = i

        # Build keyword index
        text = item.get('description', '').lower()
        words = text.split()
        for word in words:
            if len(word) > 3:  # Skip short words
                if word not in index['keyword_index']:
                    index['keyword_index'][word] = []
                index['keyword_index'][word].append(i)

    return index


def build_semantic_index(nvd_data: List[Dict], model_name: str = 'all-MiniLM-L6-v2') -> Dict:
    """Build semantic search index using sentence transformers"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Falling back to simple index...")
        return build_simple_index(nvd_data)

    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # Build corpus
    corpus = build_text_corpus(nvd_data)

    # Encode corpus
    print("Encoding corpus...")
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index if available
    if FAISS_AVAILABLE:
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings.astype('float32'))

        index = {
            'type': 'semantic_faiss',
            'faiss_index': faiss_index,
            'embeddings': embeddings,
            'model_name': model_name,
            'metadata': nvd_data,
            'corpus': corpus
        }
    else:
        # Use simple numpy-based search
        index = {
            'type': 'semantic_numpy',
            'embeddings': embeddings,
            'model_name': model_name,
            'metadata': nvd_data,
            'corpus': corpus
        }

    return index


def search_index(index: Dict, query: str, top_k: int = 3) -> List[Dict]:
    """
    Search the index and return top-K results

    This is a helper function for testing the index
    """
    if index['type'] == 'simple':
        # Keyword-based search
        query_words = query.lower().split()
        scores = {}

        for word in query_words:
            if word in index['keyword_index']:
                for doc_idx in index['keyword_index'][word]:
                    scores[doc_idx] = scores.get(doc_idx, 0) + 1

        # Get top-K
        top_indices = sorted(scores.keys(), key=lambda x: -scores[x])[:top_k]
        return [index['metadata'][i] for i in top_indices]

    elif 'semantic' in index['type']:
        # Semantic search
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(index['model_name'])

        query_embedding = model.encode([query], convert_to_numpy=True)

        if index['type'] == 'semantic_faiss':
            distances, indices = index['faiss_index'].search(query_embedding.astype('float32'), top_k)
            return [index['metadata'][i] for i in indices[0]]
        else:
            # Numpy-based search
            similarities = np.dot(index['embeddings'], query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [index['metadata'][i] for i in top_indices]

    return []


def main():
    parser = argparse.ArgumentParser(description="Build retrieval index for RAG")
    parser.add_argument('--nvd-data', type=str, required=True, help='NVD metadata JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file')
    parser.add_argument('--index-type', type=str, default='semantic', choices=['simple', 'semantic'],
                       help='Index type (simple or semantic)')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')

    args = parser.parse_args()

    # Load NVD data
    print("Loading NVD metadata...")
    nvd_file = Path(args.nvd_data)
    nvd_data = load_nvd_metadata(nvd_file)
    print(f"Loaded {len(nvd_data)} CVE entries")

    # Build index
    print(f"\nBuilding {args.index_type} index...")
    if args.index_type == 'semantic':
        index = build_semantic_index(nvd_data, args.model)
    else:
        index = build_simple_index(nvd_data)
        index['type'] = 'simple'

    # Save index
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving index to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(index, f)

    print(f"\n✓ Index built successfully!")
    print(f"  Type: {index['type']}")
    print(f"  Entries: {len(nvd_data)}")

    # Test search
    print("\nTesting search with sample query...")
    test_query = "Apache Log4j remote code execution"
    results = search_index(index, test_query, top_k=3)

    print(f"\nTop 3 results for '{test_query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('cve_id', 'N/A')}: {result.get('description', 'N/A')[:100]}...")


if __name__ == '__main__':
    main()
