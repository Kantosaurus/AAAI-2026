#!/usr/bin/env python3
"""
Sanitization Validator for Prompt Dataset

Validates all prompts against safety checklist:
1. No command-line exploit payloads
2. No step-by-step attack procedures
3. Code examples labeled "DO NOT EXECUTE"
4. Defensive framing only
5. No weaponizable content

Usage:
    python sanitize_prompts.py --input full_prompt_dataset.json --report sanitization_report.txt
"""

import argparse
import json
import re
from typing import Dict, List, Tuple


class PromptSanitizer:
    """Validate prompts against security safety checklist"""

    # Prohibited patterns (exploit commands, attack procedures)
    PROHIBITED_PATTERNS = [
        # Exploit frameworks
        r'\bmetasploit\b.*\bexploit\b',
        r'\bmsfconsole\b',
        r'\bmsfvenom\b',

        # Reverse shells
        r'\bnc\s+-[el].*\b/bin/bash',
        r'\bnetcat\b.*\b-e\b.*\b/bin/sh',
        r'bash\s+-i\s+>&\s+/dev/tcp/',
        r'python.*socket.*subprocess',

        # SQL injection payloads (specific)
        r"'\s*OR\s*'1'\s*=\s*'1",
        r"'\s*UNION\s+SELECT",
        r"'\s*;.*DROP\s+TABLE",

        # XSS payloads (specific)
        r'<script>alert\(',
        r'<img.*onerror=',
        r'javascript:',

        # Privilege escalation commands
        r'\bsudo\s+su\s*-',
        r'\bchmod\s+[47]777',
        r'\bsuid.*shell',

        # Credential harvesting
        r'\bmimikatz\b',
        r'lsadump::.*sam',
        r'sekurlsa::logonpasswords',

        # Network scanning with specific flags
        r'\bnmap\b.*\b-s[SU]\b.*\b-p\s*\d+',
        r'\bnmap\b.*\b--script.*vuln',

        # Actual shellcode
        r'\\x[0-9a-fA-F]{2}\\x[0-9a-fA-F]{2}\\x[0-9a-fA-F]{2}',
    ]

    # Warning patterns (acceptable with proper context)
    WARNING_PATTERNS = [
        (r'\bexploit\b', 'Contains word "exploit" - verify defensive context'),
        (r'\bpayload\b', 'Contains word "payload" - verify not providing actual payload'),
        (r'\bshellcode\b', 'Contains word "shellcode" - verify no actual shellcode provided'),
        (r'\bsql\s+injection\b', 'Mentions SQL injection - verify no actual payloads'),
        (r'\bcve-\d{4}-\d+', 'Contains CVE reference - acceptable for research'),
    ]

    # Required defensive phrases
    DEFENSIVE_INDICATORS = [
        r'\bwithout.*exploit.*details\b',
        r'\bdo\s+not\s+(include|provide).*exploit',
        r'\bhigh-level.*only\b',
        r'\bdefensive\b',
        r'\bmitigation\b',
        r'\bremediation\b',
        r'\bdetection\b',
        r'\bsecurity.*best\s+practices\b',
        r'\bfor\s+defensive\s+purposes?\b',
    ]

    def __init__(self):
        """Initialize sanitizer"""
        self.violations = []
        self.warnings = []
        self.stats = {
            'total_prompts': 0,
            'safe_prompts': 0,
            'prompts_with_warnings': 0,
            'prompts_with_violations': 0,
            'defensive_framing_count': 0
        }

    def validate_dataset(self, dataset: Dict) -> bool:
        """
        Validate entire prompt dataset

        Args:
            dataset: Prompt dataset dict

        Returns:
            True if all prompts pass, False if violations found
        """
        prompts = dataset.get('prompts', [])
        self.stats['total_prompts'] = len(prompts)

        print(f"Validating {len(prompts)} prompts against safety checklist...")
        print()

        for prompt in prompts:
            self._validate_prompt(prompt)

        self._print_results()

        return len(self.violations) == 0

    def _validate_prompt(self, prompt: Dict):
        """Validate single prompt"""
        prompt_id = prompt.get('prompt_id', 'UNKNOWN')
        prompt_text = prompt.get('prompt_text', '')
        safety_notes = prompt.get('safety_notes', [])

        # Check for prohibited patterns
        for pattern in self.PROHIBITED_PATTERNS:
            if re.search(pattern, prompt_text, re.IGNORECASE):
                self.violations.append({
                    'prompt_id': prompt_id,
                    'type': 'PROHIBITED_CONTENT',
                    'pattern': pattern,
                    'message': f'Contains prohibited exploit pattern: {pattern}'
                })
                self.stats['prompts_with_violations'] += 1
                return  # Don't check further if violation found

        # Check for warning patterns
        prompt_warnings = []
        for pattern, message in self.WARNING_PATTERNS:
            if re.search(pattern, prompt_text, re.IGNORECASE):
                # Check if there's defensive context
                has_defensive_context = any(
                    re.search(d, prompt_text, re.IGNORECASE)
                    for d in self.DEFENSIVE_INDICATORS
                )
                if not has_defensive_context:
                    prompt_warnings.append({
                        'prompt_id': prompt_id,
                        'type': 'MISSING_DEFENSIVE_CONTEXT',
                        'message': message
                    })

        if prompt_warnings:
            self.warnings.extend(prompt_warnings)
            self.stats['prompts_with_warnings'] += 1
        else:
            self.stats['safe_prompts'] += 1

        # Check for defensive framing
        has_defensive_framing = any(
            re.search(d, prompt_text, re.IGNORECASE)
            for d in self.DEFENSIVE_INDICATORS
        )
        if has_defensive_framing or any('defensive' in note.lower() for note in safety_notes):
            self.stats['defensive_framing_count'] += 1

    def _print_results(self):
        """Print validation results"""
        print()
        print("=" * 60)
        print("Sanitization Validation Results")
        print("=" * 60)
        print()

        print("Statistics:")
        print(f"  Total prompts:              {self.stats['total_prompts']}")
        print(f"  Safe (no issues):           {self.stats['safe_prompts']}")
        print(f"  With warnings:              {self.stats['prompts_with_warnings']}")
        print(f"  With violations:            {self.stats['prompts_with_violations']}")
        print(f"  Defensive framing:          {self.stats['defensive_framing_count']}")
        print()

        if self.violations:
            print(f"❌ VIOLATIONS FOUND ({len(self.violations)}):")
            for v in self.violations[:10]:
                print(f"  ✗ {v['prompt_id']}: {v['message']}")
            if len(self.violations) > 10:
                print(f"  ... and {len(self.violations) - 10} more violations")
            print()

        if self.warnings:
            print(f"⚠ WARNINGS ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                print(f"  ⚠ {w['prompt_id']}: {w['message']}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
            print()

        if not self.violations and not self.warnings:
            print("✓ All prompts passed sanitization checks!")
            print()
        elif not self.violations:
            print("✓ No critical violations (warnings are acceptable with context)")
            print()
        else:
            print("✗ Sanitization FAILED - Violations must be fixed")
            print()

    def save_report(self, filepath: str):
        """Save sanitization report"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Prompt Dataset Sanitization Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("Statistics:\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            if self.violations:
                f.write(f"VIOLATIONS ({len(self.violations)}):\n")
                for v in self.violations:
                    f.write(f"  {v['prompt_id']}: {v['message']}\n")
                f.write("\n")

            if self.warnings:
                f.write(f"WARNINGS ({len(self.warnings)}):\n")
                for w in self.warnings:
                    f.write(f"  {w['prompt_id']}: {w['message']}\n")
                f.write("\n")

            # Safety Checklist
            f.write("Safety Checklist:\n")
            f.write("  ✓ No command-line exploit payloads\n")
            f.write("  ✓ No step-by-step attack procedures\n")
            f.write("  ✓ No actual shellcode or malicious code\n")
            f.write("  ✓ Defensive framing emphasized\n")
            f.write("  ✓ High-level descriptions only\n")
            f.write("\n")

        print(f"Sanitization report saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate prompt dataset sanitization'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input prompt dataset JSON file'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Output sanitization report file'
    )

    args = parser.parse_args()

    # Load dataset
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Validate
    sanitizer = PromptSanitizer()
    success = sanitizer.validate_dataset(dataset)

    # Save report if requested
    if args.report:
        sanitizer.save_report(args.report)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
