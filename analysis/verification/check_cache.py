#!/usr/bin/env python3
"""Quick script to check cache structure"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.run_cache import create_run_cache

# Load cache
cache = create_run_cache('SIMPLe-10000', skip_cache=False)

# Check iteration 0
print("=== Checking iteration 0 ===")
prop_cache = cache.load_proposed_questions(0)
inf_cache = cache.load_inferences(0)

if prop_cache:
    print(f"Proposed questions:")
    print(f"  Total generated: {len(prop_cache.all_generated)}")
    print(f"  Valid: {len(prop_cache.valid)}")
    print(f"  Deduplicated: {len(prop_cache.deduplicated)}")

    if len(prop_cache.deduplicated) > 0:
        sample = prop_cache.deduplicated[0]
        print(f"\nSample question:")
        print(f"  ID: {sample.question_id}")
        print(f"  is_valid: {sample.is_valid}")
        print(f"  was_solved: {sample.was_solved}")

if inf_cache:
    print(f"\nInference results:")
    print(f"  Total results: {len(inf_cache.results)}")

    if len(inf_cache.results) > 0:
        sample = inf_cache.results[0]
        print(f"\nSample inference result:")
        print(f"  ID: {sample.question_id}")
        print(f"  n_passing: {sample.n_passing}")
        print(f"  n_total: {sample.n_total}")

    # Check if IDs match
    inf_dict = inf_cache.to_dict()
    if prop_cache and len(prop_cache.deduplicated) > 0:
        sample_q = prop_cache.deduplicated[0]
        if sample_q.question_id in inf_dict:
            print(f"\n✓ Question ID found in inference cache")
        else:
            print(f"\n✗ Question ID NOT found in inference cache")
            print(f"   Question ID: {sample_q.question_id}")
            print(f"   First 5 inference IDs: {list(inf_dict.keys())[:5]}")
