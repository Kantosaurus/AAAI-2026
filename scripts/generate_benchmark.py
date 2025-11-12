#!/usr/bin/env python3
"""
Generate hallu-sec-benchmark.json from prompt templates
Extracts, sanitizes, and exports ~400 prompts with proper safety checks
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Safety sanitization checklist
UNSAFE_PATTERNS = [
    r'(?i)(execute|run|eval)\s*\(',  # Code execution
    r'(?i)nc\s+-[a-z]*e',  # Netcat reverse shells
    r'(?i)curl.*\|\s*(bash|sh)',  # Remote script execution
    r'(?i)wget.*&&',  # Download and execute chains
    r'(?i)powershell\s+-enc',  # Encoded PowerShell
    r'(?i)/bin/(bash|sh).*-c',  # Shell command injection
    r'(?i)import\s+os.*system',  # Python OS commands (if shown as executable)
    r'(?i)cmd\.exe\s+/c',  # Windows command execution
]

# Labels that should trigger "DO NOT EXECUTE" warnings
CODE_KEYWORDS = ['exploit', 'payload', 'shellcode', 'reverse shell', 'bind shell']

class PromptExtractor:
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.prompts = []
        self.prompt_id_counter = 1

    def extract_from_markdown(self, file_path: Path, category: str) -> List[Dict]:
        """Extract prompts from markdown template files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        extracted = []

        # Pattern to match prompt blocks
        # Looking for patterns like: **REAL-###** or **SYNTHETIC-###**
        prompt_pattern = r'\*\*(?P<type>REAL|SYNTHETIC|BATCH|TYPO|VALID|INVALID|RESERVED)-(?P<num>\d+)\*\*.*?\n```\n(?P<prompt>.*?)\n```'

        for match in re.finditer(prompt_pattern, content, re.DOTALL):
            prompt_type = match.group('type')
            prompt_num = match.group('num')
            prompt_text = match.group('prompt').strip()

            # Determine if this is a real or synthetic probe
            is_synthetic = prompt_type in ['SYNTHETIC', 'BATCH', 'TYPO', 'INVALID', 'RESERVED']

            # Extract ground truth from following lines
            ground_truth = self._extract_ground_truth(content, match.end())

            extracted.append({
                'id': f'{category}-{prompt_type}-{prompt_num}',
                'category': category,
                'prompt': prompt_text,
                'is_synthetic': is_synthetic,
                'ground_truth': ground_truth
            })

        return extracted

    def _extract_ground_truth(self, content: str, start_pos: int) -> Dict:
        """Extract ground truth information following a prompt"""
        # Look for the next few lines after the prompt for ground truth
        snippet = content[start_pos:start_pos+500]

        gold_truth = {
            'exists': None,
            'description': '',
            'references': []
        }

        # Check for "Ground Truth:" lines
        gt_match = re.search(r'- Ground Truth:\s*(.+?)(?:\n-|\n\*\*)', snippet, re.DOTALL)
        if gt_match:
            gt_text = gt_match.group(1).strip()
            gold_truth['description'] = gt_text

            # Determine existence
            if any(x in gt_text for x in ['DOES NOT EXIST', 'does not exist', 'NON-EXISTENT', 'non-existent']):
                gold_truth['exists'] = False
            elif any(x in gt_text for x in ['EXISTS', 'exists', 'YES', 'PUBLISHED']):
                gold_truth['exists'] = True

        # Look for CVE references
        cve_refs = re.findall(r'CVE-\d{4}-\d{4,7}', snippet)
        gold_truth['references'] = list(set(cve_refs))

        return gold_truth

    def sanitize_prompt(self, prompt: str) -> tuple[str, List[str]]:
        """
        Sanitize prompt according to safety checklist
        Returns: (sanitized_prompt, list_of_warnings)
        """
        warnings = []
        sanitized = prompt

        # Check for unsafe command patterns
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, prompt):
                warnings.append(f'Contains potentially executable command pattern')
                # Add warning label
                if 'DO NOT EXECUTE' not in sanitized:
                    sanitized += '\n\n[SAFETY NOTE: DO NOT EXECUTE ANY CODE EXAMPLES]'
                break

        # Check for exploit-related keywords that should have warnings
        for keyword in CODE_KEYWORDS:
            if keyword.lower() in prompt.lower():
                if 'do not' not in prompt.lower() or 'without' not in prompt.lower():
                    warnings.append(f'Contains "{keyword}" without explicit safety warning')
                    # Add safety framing if missing
                    if 'do not' not in sanitized.lower():
                        sanitized += f'\n\n[NOTE: Focus on defensive understanding only. Do not execute or develop exploit code.]'
                break

        # Ensure defensive framing for attack-related queries
        attack_terms = ['attack', 'exploit', 'penetrate', 'bypass']
        if any(term in prompt.lower() for term in attack_terms):
            if not any(defensive in prompt.lower() for defensive in ['defend', 'mitigate', 'protect', 'detect', 'high-level', 'without']):
                warnings.append('Contains attack terminology without defensive framing')

        return sanitized, warnings

    def load_all_templates(self) -> List[Dict]:
        """Load all prompt templates from files"""
        template_files = {
            'vulnerability_summary': self.templates_dir / 'prompts' / '01_vulnerability_summary_templates.md',
            'cve_existence': self.templates_dir / 'prompts' / '02_cve_existence_lookup_templates.md',
            'malware_description': self.templates_dir / 'prompts' / '03_malware_family_description_templates.md',
            'secure_configuration': self.templates_dir / 'prompts' / '04_secure_configuration_templates.md',
            'pentest_reporting': self.templates_dir / 'prompts' / '05_pentest_reporting_templates.md',
        }

        all_prompts = []

        for category, file_path in template_files.items():
            if file_path.exists():
                prompts = self.extract_from_markdown(file_path, category)
                print(f'Extracted {len(prompts)} prompts from {category}')
                all_prompts.extend(prompts)

        return all_prompts

    def select_subset(self, all_prompts: List[Dict], target_real: int = 250, target_synthetic: int = 150) -> List[Dict]:
        """Select a representative subset of prompts"""
        # Separate by type
        real_prompts = [p for p in all_prompts if not p['is_synthetic']]
        synthetic_prompts = [p for p in all_prompts if p['is_synthetic']]

        print(f'Available: {len(real_prompts)} real, {len(synthetic_prompts)} synthetic')

        # If we have more than target, select evenly across categories
        selected_real = self._select_evenly(real_prompts, target_real)
        selected_synthetic = self._select_evenly(synthetic_prompts, target_synthetic)

        return selected_real + selected_synthetic

    def _select_evenly(self, prompts: List[Dict], target: int) -> List[Dict]:
        """Select prompts evenly across categories"""
        if len(prompts) <= target:
            return prompts

        # Group by category
        by_category = {}
        for p in prompts:
            cat = p['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(p)

        # Calculate per-category quota
        per_category = target // len(by_category)
        remainder = target % len(by_category)

        selected = []
        for i, (cat, cat_prompts) in enumerate(sorted(by_category.items())):
            quota = per_category + (1 if i < remainder else 0)
            selected.extend(cat_prompts[:quota])

        return selected[:target]  # Ensure we don't exceed target

    def generate_benchmark(self, output_path: Path):
        """Generate the final hallu-sec-benchmark.json file"""
        # Load all prompts
        all_prompts = self.load_all_templates()
        print(f'Total prompts extracted: {len(all_prompts)}')

        # Select subset (~400 prompts)
        selected_prompts = self.select_subset(all_prompts, target_real=250, target_synthetic=150)
        print(f'Selected {len(selected_prompts)} prompts for benchmark')

        # Sanitize and format for export
        benchmark = {
            'metadata': {
                'version': '1.0',
                'generated': datetime.now().isoformat(),
                'total_prompts': len(selected_prompts),
                'real_prompts': len([p for p in selected_prompts if not p['is_synthetic']]),
                'synthetic_prompts': len([p for p in selected_prompts if p['is_synthetic']]),
                'categories': list(set(p['category'] for p in selected_prompts)),
                'safety_note': 'All prompts have been sanitized. This dataset is for DEFENSIVE RESEARCH ONLY.',
            },
            'prompts': []
        }

        sanitization_warnings = []

        for i, prompt_data in enumerate(selected_prompts, 1):
            # Sanitize
            sanitized_prompt, warnings = self.sanitize_prompt(prompt_data['prompt'])

            if warnings:
                sanitization_warnings.append({
                    'id': prompt_data['id'],
                    'warnings': warnings
                })

            # Format for export
            benchmark['prompts'].append({
                'id': f'prompt_{i:04d}',
                'original_id': prompt_data['id'],
                'category': prompt_data['category'],
                'prompt': sanitized_prompt,
                'is_synthetic_probe': prompt_data['is_synthetic'],
                'gold_label': {
                    'exists': prompt_data['ground_truth'].get('exists'),
                    'description': prompt_data['ground_truth'].get('description', ''),
                },
                'gold_refs': prompt_data['ground_truth'].get('references', []),
            })

        # Write benchmark file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark, f, indent=2, ensure_ascii=False)

        print(f'\nBenchmark saved to: {output_path}')
        print(f'Total prompts: {len(benchmark["prompts"])}')
        print(f'Real/grounded: {benchmark["metadata"]["real_prompts"]}')
        print(f'Synthetic/probes: {benchmark["metadata"]["synthetic_prompts"]}')

        # Save sanitization report
        if sanitization_warnings:
            report_path = output_path.parent / 'sanitization_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_warnings': len(sanitization_warnings),
                    'warnings': sanitization_warnings
                }, f, indent=2)
            print(f'\nSanitization warnings: {len(sanitization_warnings)} (see {report_path})')
        else:
            print('\nNo sanitization warnings detected.')

        return benchmark


if __name__ == '__main__':
    # Set paths
    repo_root = Path(__file__).parent.parent
    output_path = repo_root / 'data' / 'prompts' / 'hallu-sec-benchmark.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate benchmark
    extractor = PromptExtractor(repo_root)
    benchmark = extractor.generate_benchmark(output_path)

    print('\n✓ Benchmark generation complete!')
    print(f'✓ Output: {output_path}')
    print(f'✓ Ready for Phase B pilot runs')
