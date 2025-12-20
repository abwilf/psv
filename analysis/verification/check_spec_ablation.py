#!/usr/bin/env python3
"""Check if -spec ablation records verification results"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.run_cache import create_run_cache

# Load cache
cache = create_run_cache('SIMPLe-specver', skip_cache=False)

# Check iteration 0
print("=== Checking SIMPLe-specver iteration 0 ===")
prop_cache = cache.load_proposed_questions(0)

if prop_cache:
    print(f"Total generated: {len(prop_cache.all_generated)}")
    print(f"Valid: {len(prop_cache.valid)}")

    # Check is_valid field distribution
    valid_count = sum(1 for q in prop_cache.all_generated if q.is_valid)
    invalid_count = sum(1 for q in prop_cache.all_generated if not q.is_valid)

    print(f"\nis_valid distribution:")
    print(f"  True: {valid_count}")
    print(f"  False: {invalid_count}")

    # Sample some questions
    print(f"\nFirst 5 questions:")
    for i, q in enumerate(prop_cache.all_generated[:5]):
        print(f"  {i+1}. is_valid={q.is_valid}, validation_error={q.validation_error}")
