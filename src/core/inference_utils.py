"""Inference utilities for pass@k evaluation with varied few-shot prompts."""
import itertools
import numpy as np
from src.core.data_utils import Timer
from typing import List, Dict, Callable, Tuple, Any
from src.core.evaluation import run_codes, run_specs, K_LIST
from src.core.prompt_utils import postprocess_verus

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array. From https://github.com/huggingface/evaluate/blob/b3820eb820702611cd0c2247743d764f2a7fe916/metrics/code_eval/code_eval.py"""
    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def inference_and_passk(
    all_msgs: List[List[Dict[str, str]]],
    sampling_params: Dict[str, Any],
    question_idxs: np.ndarray,
    pass_fn: Callable[[List[str]], List[bool]],
    args: Any,
    inf_bs: int,
    return_format: str = 'cache_results',
    postprocessed_responses: List[str] = None  # Optional: for storing processed completions
) -> Any:
    """
    Run inference with pass@k evaluation using varied prompts.

    This function supports pass@k evaluation where each question gets k different
    inference attempts (potentially with varied few-shot prompts). It groups responses
    by question, evaluates them, and computes pass@k metrics.

    Args:
        all_msgs: List of message lists (prompts), already constructed with varied few-shot examples.
                  Length = total number of inference calls (n_questions * k)
        sampling_params: Dict with model, temperature, max_tokens, etc. Will use n=1 for each.
        question_idxs: Array mapping each msg index to its question ID.
                       Example: [0,0,0,1,1,1,2,2,2] means 3 questions with k=3 attempts each.
        pass_fn: Function that takes List[str] of responses and returns List[bool] of pass/fail.
                 IMPORTANT: For completions, responses must be in same order as all_msgs to match specs!
        args: Command line arguments (needs sglang_server_name, debug_mode, etc.)
        inf_bs: Inference batch size
        return_format: 'cache_results' for completion evaluation (returns list of dicts),
                       'tuple' for spec evaluation (returns (responses, passed))

    Returns:
        If return_format='cache_results': List of cache_result dicts (one per question)
        If return_format='tuple': (all_responses, passed) where all_responses is array of
                                  best response per question, passed is bool array
    """
    from inference import inference
    # Run inference with n=1 for each prompt
    sampling_params = dict(sampling_params)  # Don't modify caller's dict
    sampling_params['n'] = 1

    print(f"   🔄 Running inference for {len(all_msgs)} prompts ({len(np.unique(question_idxs))} questions)")
    raw_results = inference(all_msgs, sampling_params, inf_bs, args)

    # Extract response strings
    responses = [r.choices[0].message.content for r in raw_results]

    # Evaluate all responses
    with Timer(f"   📝 Got {len(responses)} responses, evaluating...", level=3):
        passed = pass_fn(responses)
    passed = np.array(passed, dtype=bool)

    # Group responses by question
    n_questions = len(np.unique(question_idxs))
    question_idxs = np.array(question_idxs)

    # Determine k (attempts per question)
    k = len(all_msgs) // n_questions
    assert len(all_msgs) == n_questions * k, \
        f"all_msgs length {len(all_msgs)} not divisible by n_questions {n_questions}"

    # Reshape responses and passed arrays by question
    responses_by_q = np.array(responses).reshape(n_questions, k)
    passed_by_q = passed.reshape(n_questions, k)

    # If postprocessed_responses provided, use those for storage instead
    if postprocessed_responses is not None:
        responses_for_storage = np.array(postprocessed_responses).reshape(n_questions, k)
    else:
        responses_for_storage = responses_by_q

    if return_format == 'cache_results':
        return _format_as_cache_results(responses_for_storage, passed_by_q, k)
    elif return_format == 'tuple':
        return _format_as_tuple(responses_for_storage, passed_by_q)
    else:
        raise ValueError(f"Unknown return_format: {return_format}")


def _format_as_cache_results(responses_by_q: np.ndarray, passed_by_q: np.ndarray, k: int) -> List[Dict]:
    """
    Format results as cache_result dicts for database storage.

    Returns list of dicts with fields:
    - pass_rate, n_passing, n_total, passing_solution, failing_solution
    - all_solutions, pass_at_k (dict), solved (bool)
    """
    cache_results = []
    n_questions = len(responses_by_q)

    sol_at_k = {}
    for _k in K_LIST:
        if _k <= k:
            # Use unbiased estimator for pass@k
            num_correct = passed_by_q.sum(axis=1)  # Number of correct per question
            sol_at_k[f"sol@{_k}_solved"] = estimate_pass_at_k(k, num_correct, _k)

    for i in range(n_questions):
        n_passing = int(passed_by_q[i].sum())
        n_total = k
        pass_rate = n_passing / n_total if n_total > 0 else 0.0

        # Get best response (first passing one, or first failing one if none pass)
        has_passing = passed_by_q[i].any()
        if has_passing:
            first_pass_idx = passed_by_q[i].argmax()  # Index of first True
            passing_solution = responses_by_q[i, first_pass_idx]
            failing_solution = ""
        else:
            passing_solution = ""
            failing_solution = responses_by_q[i, 0]

        # Create pass@k dict for this question
        pass_at_k_dict = {}
        for k_str, values in sol_at_k.items():
            pass_at_k_dict[k_str] = float(values[i])

        cache_result = {
            'pass_rate': pass_rate,
            'n_passing': n_passing,
            'n_total': n_total,
            'passing_solution': passing_solution,
            'failing_solution': failing_solution,
            'all_solutions': responses_by_q[i].tolist(),
            'pass_at_k': pass_at_k_dict,
            'solved': bool(has_passing)
        }
        cache_results.append(cache_result)

    # Print summary
    n_solved = sum(cr['solved'] for cr in cache_results)
    print(f"📊 PASS@{k} RESULTS:")
    print(f"   Passed: {n_solved} / {n_questions} = {n_solved/n_questions:.2%}")
    for _k in sorted([int(key.split('@')[1].split('_')[0]) for key in sol_at_k.keys()]):
        if _k <= k:
            rate = sol_at_k[f"sol@{_k}_solved"].mean()
            print(f"   Pass@{_k}: {rate:.2%}")

    return cache_results


def _format_as_tuple(responses_by_q: np.ndarray, passed_by_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Format results as (all_responses, passed) tuple for spec generation use case.

    Returns:
    - all_responses: Array of best response per question (first passing, or first if none pass)
    - passed: Boolean array indicating if each question has at least one passing response
    """
    n_questions = len(responses_by_q)
    k = responses_by_q.shape[1]

    # Get best response for each question
    all_responses = []
    all_passed = []

    for i in range(n_questions):
        has_passing = passed_by_q[i].any()
        if has_passing:
            first_pass_idx = passed_by_q[i].argmax()
            all_responses.append(responses_by_q[i, first_pass_idx])
        else:
            all_responses.append(responses_by_q[i, 0])
        all_passed.append(has_passing)

    all_responses = np.array(all_responses)
    all_passed = np.array(all_passed, dtype=bool)

    # Print summary
    print(f'\n============\nPass@{k}: {sum(all_passed)} / {len(all_passed)} = {sum(all_passed)/len(all_passed):.2%}\n============\n')

    return all_responses, all_passed


def make_completion_pass_fn(specs: List[str], k: int, postprocessed_out: List[str] = None) -> Callable[[List[str]], List[bool]]:
    """
    Create pass_fn for Verus code completion evaluation.

    This function creates a closure that has the specs and k value, and returns
    a pass_fn that can be used with inference_and_passk.

    Args:
        specs: List of spec strings (one per question)
        k: Number of attempts per question
        postprocessed_out: Optional list to populate with postprocessed completions for storage

    Returns:
        pass_fn that takes responses and returns boolean array of pass/fail
    """
    def pass_fn(responses: List[str]) -> List[bool]:
        """Evaluate completions by combining with specs and running through Verus."""
        # Postprocess and combine with specs
        assert len(specs)==len(responses)
        codes = []
        postprocessed_completions = []
        for spec, response in zip(specs, responses):
            completion = postprocess_verus(spec, response)
            postprocessed_completions.append(completion)
            code = spec + '\n' + completion
            codes.append(code)

        # Store postprocessed completions if output list provided
        if postprocessed_out is not None:
            postprocessed_out.clear()
            postprocessed_out.extend(postprocessed_completions)

        # Run through Verus evaluation
        passed = run_codes(codes, quiet=False)
        return passed

    return pass_fn


def make_spec_pass_fn(skip_verification: bool = False) -> Callable[[List[str]], List[bool]]:
    """
    Create pass_fn for Verus spec evaluation.

    Args:
        skip_verification: If True, skip verification and accept all specs (for ablation studies)

    Returns:
        pass_fn that takes spec responses and returns boolean array of pass/fail
    """
    def pass_fn(responses: List[str]) -> List[bool]:
        """Evaluate specs by running through Verus spec verification."""
        if skip_verification:
            # Accept all specs without verification (for -specver ablation)
            print(f"⚠️  -specver ablation: Skipping spec verification, accepting all {len(responses)} specs")
            return [True] * len(responses)

        # run_specs returns: (all_eval_results, syntax_pass_results, runnable_specs, cleaned_specs)
        syntax_pass_results = run_specs(responses, quiet=False)[1]
        return syntax_pass_results

    return pass_fn


# ============================================================================
# Unit Tests
# ============================================================================

def test_format_as_cache_results():
    """Test cache_results formatting."""
    print("Testing _format_as_cache_results...")

    # 2 questions, k=3 attempts each
    responses_by_q = np.array([
        ['resp0_0', 'resp0_1', 'resp0_2'],  # Q0: passes on attempt 1
        ['resp1_0', 'resp1_1', 'resp1_2'],  # Q1: never passes
    ])
    passed_by_q = np.array([
        [False, True, False],  # Q0: passes on attempt 1
        [False, False, False],  # Q1: never passes
    ])

    results = _format_as_cache_results(responses_by_q, passed_by_q, k=3)

    # Check Q0 (has passing solution: 1 out of 3 attempts pass)
    assert results[0]['solved'] == True
    assert results[0]['passing_solution'] == 'resp0_1'
    assert results[0]['failing_solution'] == ''
    assert results[0]['n_passing'] == 1
    assert results[0]['n_total'] == 3
    assert results[0]['pass_rate'] == 1/3
    # Unbiased pass@1 with n=3, c=1: 1 - C(2,1)/C(3,1) = 1 - 2/3 = 1/3
    assert abs(results[0]['pass_at_k']['sol@1_solved'] - 1/3) < 0.001

    # Check Q1 (no passing solution)
    assert results[1]['solved'] == False
    assert results[1]['passing_solution'] == ''
    assert results[1]['failing_solution'] == 'resp1_0'
    assert results[1]['n_passing'] == 0
    assert results[1]['n_total'] == 3

    print("✅ _format_as_cache_results tests passed")


def test_format_as_tuple():
    """Test tuple formatting."""
    print("Testing _format_as_tuple...")

    responses_by_q = np.array([
        ['resp0_0', 'resp0_1', 'resp0_2'],
        ['resp1_0', 'resp1_1', 'resp1_2'],
    ])
    passed_by_q = np.array([
        [False, True, False],
        [False, False, False],
    ])

    all_responses, all_passed = _format_as_tuple(responses_by_q, passed_by_q)

    assert len(all_responses) == 2
    assert len(all_passed) == 2
    assert all_responses[0] == 'resp0_1'  # First passing
    assert all_responses[1] == 'resp1_0'  # None pass, so first one
    assert all_passed[0] == True
    assert all_passed[1] == False

    print("✅ _format_as_tuple tests passed")


def test_make_completion_pass_fn():
    """Test completion pass function creation."""
    print("Testing make_completion_pass_fn...")

    # Mock specs
    specs = ['spec0', 'spec1']
    k = 2

    # Create pass_fn
    pass_fn = make_completion_pass_fn(specs, k)

    # Test that it's callable (we can't fully test without mocking run_codes)
    assert callable(pass_fn)

    print("✅ make_completion_pass_fn tests passed")


def test_make_spec_pass_fn():
    """Test spec pass function creation."""
    print("Testing make_spec_pass_fn...")

    # Create pass_fn
    pass_fn = make_spec_pass_fn()

    # Test that it's callable
    assert callable(pass_fn)

    print("✅ make_spec_pass_fn tests passed")


if __name__ == '__main__':
    print("Running inference_utils unit tests...\n")
    test_format_as_cache_results()
    test_format_as_tuple()
    test_make_completion_pass_fn()
    test_make_spec_pass_fn()
    print("\n✅ All inference_utils unit tests passed!")
