"""I/O utilities for loading and saving data"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_json_file(file_path: Union[str, Path]) -> Any:
    """
    Load JSON file with error handling

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading JSON file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.debug(f"Successfully loaded {file_path}")
    return data


def save_json_file(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with error handling

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving JSON file: {file_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

    logger.debug(f"Successfully saved {file_path}")


def load_pilot_results(results_dir: Union[str, Path]) -> List[Dict]:
    """
    Load all pilot result JSON files from a directory

    Args:
        results_dir: Directory containing pilot_*.json files

    Returns:
        List of all results combined
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    logger.info(f"Loading pilot results from: {results_dir}")

    all_results = []
    json_files = list(results_dir.glob("pilot_*.json"))

    if not json_files:
        logger.warning(f"No pilot_*.json files found in {results_dir}")
        return []

    for json_file in json_files:
        logger.debug(f"Loading {json_file.name}")
        data = load_json_file(json_file)

        # Handle different formats
        if isinstance(data, list):
            all_results.extend(data)
        elif isinstance(data, dict) and 'results' in data:
            all_results.extend(data['results'])
        else:
            logger.warning(f"Unexpected format in {json_file.name}, skipping")

    logger.info(f"Loaded {len(all_results)} total results from {len(json_files)} files")
    return all_results


def load_annotations_csv(csv_file: Union[str, Path]) -> Dict[tuple, Dict]:
    """
    Load annotations from CSV file

    Args:
        csv_file: Path to annotations CSV

    Returns:
        Dict mapping (prompt_id, model) to annotation data
    """
    csv_file = Path(csv_file)

    if not csv_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {csv_file}")

    logger.info(f"Loading annotations from: {csv_file}")

    annotations = {}

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['prompt_id'], row['model'])
            annotations[key] = {
                'hallucination_binary': int(row.get('hallucination_binary', 0)),
                'hallucination_types': row.get('hallucination_types', ''),
                'severity': row.get('severity', ''),
                'citation_correctness': row.get('citation_correctness', ''),
                'notes': row.get('notes', '')
            }

    logger.info(f"Loaded {len(annotations)} annotations")
    return annotations


def load_multiple_result_files(file_patterns: List[str]) -> List[Dict]:
    """
    Load multiple result files by pattern

    Args:
        file_patterns: List of file paths or glob patterns

    Returns:
        Combined list of all results
    """
    all_results = []

    for pattern in file_patterns:
        pattern_path = Path(pattern)

        # If it's a glob pattern
        if '*' in pattern:
            parent = pattern_path.parent
            glob_pattern = pattern_path.name
            files = list(parent.glob(glob_pattern))
        else:
            files = [pattern_path] if pattern_path.exists() else []

        for file_path in files:
            logger.debug(f"Loading {file_path}")
            data = load_json_file(file_path)

            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict) and 'results' in data:
                all_results.extend(data['results'])

    logger.info(f"Loaded {len(all_results)} results from {len(file_patterns)} patterns")
    return all_results
