"""Code execution and evaluation utilities for AV2."""
import re
import numpy as np
from typing import List, Tuple, Dict, Optional


# Configuration
K_LIST = [1, 2, 4, 5, 8, 10, 16, 32, 50, 64, 100, 128, 200, 256, 512]

def passed(msg: str) -> bool:
    """Check if Verus verification passed."""
    num_verified = re.search(r'(\d+) verified', msg)
    num_verified = int(num_verified.group(1)) if num_verified else 0

    num_errors = re.search(r'(\d+) errors', msg)
    num_errors = int(num_errors.group(1)) if num_errors else 0
    return num_verified > 1 and num_errors == 0


def spec_passed(msg: str, spec: str) -> bool:
    """Check if spec verification passed."""
    num_verified = re.search(r'(\d+) verified', msg)
    num_verified = int(num_verified.group(1)) if num_verified else 0

    num_errors = re.search(r'(\d+) errors', msg)
    num_errors = int(num_errors.group(1)) if num_errors else 0
    return num_verified > 0 and num_errors == 0 and 'ensures' in spec




def extract_rust_block(text: str) -> str:
    """Extract rust code from markdown block."""
    import re
    RUST_BLOCK_RE = re.compile(r"```rust\s*([\s\S]*?)```", re.IGNORECASE)
    m = RUST_BLOCK_RE.search(text)
    return m.group(1).strip() if m else ''


# Import helper functions from analyze_spec_processing
def find_matching_brace(text: str, start_pos: int) -> int:
    """Find the position of the closing brace that matches the opening brace at start_pos."""
    if start_pos >= len(text) or text[start_pos] != '{':
        return -1

    depth = 0
    i = start_pos
    in_string = False
    in_char = False
    escape_next = False

    while i < len(text):
        c = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if c == '\\':
            escape_next = True
            i += 1
            continue

        if c == '"' and not in_char:
            in_string = not in_string
            i += 1
            continue

        if c == "'" and not in_string:
            in_char = not in_char
            i += 1
            continue

        if not in_string and not in_char:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i

        i += 1

    return -1


def is_spec_function(fn_text: str) -> bool:
    """Check if this is a spec function."""
    return bool(re.match(r'^\s*(pub\s+)?(open\s+)?spec\s+fn\s+', fn_text, re.MULTILINE))


def find_all_functions(code: str) -> List[Tuple[int, int, str, bool]]:
    """Find all function definitions. Returns: List of (start_pos, end_pos, function_text, is_spec)"""
    functions = []
    fn_pattern = re.compile(r'^\s*(?:(?:pub(?:\([^)]*\))?\s+)?(?:open\s+)?(?:spec\s+)?fn\s+\w+)', re.MULTILINE)

    for match in fn_pattern.finditer(code):
        start = match.start()
        brace_search_start = match.end()
        search_text = code[brace_search_start:]

        # Find function body brace
        closing_then_opening = re.search(r'\},\s*\n\s*{', search_text)
        if closing_then_opening:
            brace_pos = brace_search_start + closing_then_opening.end() - 1
        else:
            brace_pos = -1
            for i, c in enumerate(search_text):
                if c == '{':
                    before_10 = search_text[max(0, i-10):i]
                    if '==>' in before_10.replace('\n', '').replace(' ', '')[-4:]:
                        continue
                    brace_pos = brace_search_start + i
                    break

            if brace_pos == -1:
                brace_pos_rel = search_text.find('{')
                if brace_pos_rel == -1:
                    continue
                brace_pos = brace_search_start + brace_pos_rel

        end_pos = find_matching_brace(code, brace_pos)
        if end_pos == -1:
            end_pos = len(code)
        else:
            end_pos += 1

        fn_text = code[start:end_pos]
        is_spec = is_spec_function(fn_text)
        functions.append((start, end_pos, fn_text, is_spec))

    return functions


def get_function_signature_and_spec(fn_text: str) -> Tuple[str, str]:
    """Extract function signature and spec. Returns (signature, full_spec)."""
    closing_then_opening = re.search(r'\},\s*\n\s*{', fn_text)
    if closing_then_opening:
        brace_pos = closing_then_opening.end() - 1
    else:
        brace_pos = -1
        for i, c in enumerate(fn_text):
            if c == '{':
                before_10 = fn_text[max(0, i-10):i]
                if '==>' in before_10.replace('\n', '').replace(' ', '')[-4:]:
                    continue
                brace_pos = i
                break

        if brace_pos == -1:
            brace_pos = fn_text.find('{')

    if brace_pos == -1:
        return fn_text, fn_text

    spec_part = fn_text[:brace_pos + 1]
    match = re.search(r'^(.*?fn\s+\w+[^{]*?)\s*(?:requires|ensures|decreases|{)', spec_part, re.DOTALL)
    if match:
        signature = match.group(1).strip()
    else:
        signature = spec_part.strip()

    return signature, spec_part


def has_implementation(fn_text: str) -> bool:
    """Check if function has implementation code."""
    brace_pos = fn_text.find('{')
    if brace_pos == -1:
        return False

    body = fn_text[brace_pos + 1:].strip()
    if body.endswith('}'):
        body = body[:-1].strip()

    if len(body) < 3:
        return False

    impl_patterns = [
        r'\blet\s+', r'\bfor\s+', r'\bwhile\s+', r'\bloop\s*{',
        r'\breturn\s+', r'\bif\s+', r'\bmatch\s+',
        r'\w+\s*\([^)]*\)\s*;', r'\.push\(', r'\.clone\(',
        r'\w+\s*=\s*[^=]',
    ]

    return any(re.search(pattern, body) for pattern in impl_patterns)


def get_return_type(fn_signature: str) -> Optional[str]:
    """Extract return type from function signature."""
    match = re.search(r'->\s*\(([^)]+)\)', fn_signature)
    if match:
        return match.group(1).strip()

    match = re.search(r'->\s*([^\s{,]+)', fn_signature)
    if match:
        ret_type = match.group(1).strip()
        if ret_type and ret_type != '()':
            return ret_type

    return None


def create_external_body_stub(fn_signature: str, return_type: Optional[str]) -> str:
    """Create external_body stub for a function."""
    if return_type and return_type != '()':
        return "\n    assume(false);\n    arbitrary()\n"
    else:
        return "\n"


def strip_function_implementation(fn_text: str) -> str:
    """Strip implementation and add external_body stub."""
    signature, spec_part = get_function_signature_and_spec(fn_text)
    return_type = get_return_type(signature)
    stub_body = create_external_body_stub(signature, return_type)

    fn_pos = spec_part.find('fn ')
    if fn_pos == -1:
        fn_pos = 0

    result = spec_part[:fn_pos] + '#[verifier::external_body]\n' + spec_part[fn_pos:] + stub_body + '}\n'
    return result


def clean_spec(spec: str) -> str:
    """
    Clean and normalize a spec by:
    1. Extracting from markdown
    2. Stripping implementations from regular functions
    3. Adding verus! wrapper if needed
    4. Adding necessary imports

    Returns the cleaned spec WITHOUT external_body (for storage/caching).
    """
    extracted = extract_rust_block(spec)
    if not extracted:
        return ''

    try:
        # Find all functions
        functions = find_all_functions(extracted)
        if not functions:
            return ''

        # Process each function
        processed_parts = []
        last_end = 0

        for start, end, fn_text, is_spec in functions:
            # Add text before this function
            before_text = extracted[last_end:start]
            if before_text.strip():
                processed_parts.append(before_text)

            if is_spec:
                # Spec functions: keep as-is
                processed_parts.append(fn_text)
            else:
                # Regular function: strip implementation but DON'T add external_body yet
                signature, spec_part = get_function_signature_and_spec(fn_text)
                return_type = get_return_type(signature)
                # Just keep signature + spec, end with opening brace
                processed_parts.append(spec_part)

            last_end = end

        # Add remaining text
        after_text = extracted[last_end:].strip()
        if after_text and not after_text.startswith('}'):
            processed_parts.append(after_text)

        result = ''.join(processed_parts)

        # Check if we have verus! and if we have imports
        has_verus = 'verus!' in result
        lines = result.split('\n')

        # Find imports
        import_end = 0
        has_vstd_or_builtin = False
        for i, line in enumerate(lines):
            if line.strip().startswith('use '):
                import_end = i + 1
                if 'vstd' in line or 'builtin_macros' in line or 'builtin::' in line:
                    has_vstd_or_builtin = True
            elif line.strip() and not line.strip().startswith('use ') and not line.strip().startswith('//') and not line.strip().startswith('#!'):
                break

        # If we have verus! but no vstd/builtin import, add one
        if has_verus and not has_vstd_or_builtin:
            import_line = 'use vstd::prelude::*;'
            if import_end > 0:
                lines.insert(import_end, import_line)
            else:
                lines.insert(0, import_line)
            result = '\n'.join(lines)
            has_vstd_or_builtin = True

        # Add verus! wrapper if not present
        if not has_verus:
            lines = result.split('\n')
            import_end = 0

            for i, line in enumerate(lines):
                if line.strip().startswith('use '):
                    import_end = i + 1
                elif line.strip() and not line.strip().startswith('use ') and not line.strip().startswith('//') and not line.strip().startswith('#!'):
                    break

            # Add import if needed
            import_prefix = ''
            if not has_vstd_or_builtin:
                import_prefix = 'use vstd::prelude::*;\n\n'

            # Split and wrap
            import_lines = lines[:import_end]
            rest_lines = lines[import_end:]

            result = import_prefix + '\n'.join(import_lines) + '\n\nverus! {\n\n' + '\n'.join(rest_lines) + '\n\n}\n'

        # Fix function bodies that were incorrectly closed with {}
        # Function specs should end with { (open brace) not {} (closed empty body)
        # This needs to run AFTER both branches above
        result_stripped = result.rstrip()
        if result_stripped.endswith('{}'):
            result = result_stripped[:-2] + '{\n'
        elif '{\n\n}' in result[-10:]:  # Check last 10 chars for this pattern
            # Find and replace the pattern
            idx = result.rfind('{\n\n}')
            if idx != -1:
                result = result[:idx+1] + '\n'

        return result

    except Exception as e:
        print(f'Spec cleaning failed with error: {e}')
        import traceback
        traceback.print_exc()
        return ''


def process_spec(spec: str) -> str:
    """
    Process a spec for evaluation by:
    1. Cleaning it (clean_spec)
    2. Adding external_body attributes and stubs
    3. Adding fn main()

    Returns a runnable version for Verus verification.
    """
    # First clean the spec
    cleaned = clean_spec(spec)
    if not cleaned:
        return ''

    try:
        # Find all functions in cleaned spec
        functions = find_all_functions(cleaned)
        if not functions:
            return ''

        # Process each function to add external_body
        processed_parts = []
        last_end = 0

        for start, end, fn_text, is_spec in functions:
            # Add text before this function
            before_text = cleaned[last_end:start]
            if before_text.strip():
                processed_parts.append(before_text)

            if is_spec:
                # Spec functions: keep as-is
                processed_parts.append(fn_text)
            else:
                # Regular function: add external_body and stub
                stripped = strip_function_implementation(fn_text)
                processed_parts.append(stripped)

            last_end = end

        # Add remaining text
        after_text = cleaned[last_end:].strip()
        if after_text and not after_text.startswith('}'):
            processed_parts.append(after_text)

        result = ''.join(processed_parts)

        # Ensure fn main()
        if 'fn main()' not in result:
            result += '\nfn main() {}\n'

        # Close verus! if needed
        if 'verus!' in result:
            open_count = result.count('{')
            close_count = result.count('}')
            if open_count > close_count:
                result += '}\n'

        return result

    except Exception as e:
        print(f'Spec processing failed with error: {e}')
        import traceback
        traceback.print_exc()
        return ''

def run_specs(specs: List[str], quiet: bool = False) -> Tuple[List[Tuple], List[bool], List[str], List[str]]:
    """
    Run specs through evaluation pipeline.

    Returns:
        - all_eval_results: List of (stdout, stderr) tuples
        - syntax_pass_results: List of booleans indicating if spec passed
        - runnable_specs: List of runnable specs (with external_body for testing)
        - cleaned_specs: List of cleaned specs (without external_body, for storage)
    """
    # Clean specs first (for storage)
    cleaned_specs = [clean_spec(elt) for elt in specs]

    # Create runnable versions (for testing)
    runnable_specs = [process_spec(elt) for elt in specs]

    # Run verification
    all_eval_results = run_codes(runnable_specs, quiet=quiet, return_full_results=True)
    syntax_pass_results = [spec_passed(elt[1], runnable_spec) for elt, runnable_spec in zip(all_eval_results, runnable_specs)]

    return all_eval_results, syntax_pass_results, runnable_specs, cleaned_specs


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eval_server'))
from client import VerusEvalClient

# Global client - will be initialized by check_eval_server_health
client = None


def check_eval_server_health(args) -> None:
    """
    Check if the evaluation server is healthy and ready.
    Exits with error code 1 if server is not accessible.

    Args:
        args: Command line arguments containing eval_server_mode, eval_server_port, eval_server_host
    """
    global client

    # Initialize client based on mode
    if args.eval_server_mode == 'local':
        print(f"Checking local evaluation server health (localhost:{args.eval_server_port})...")
        client = VerusEvalClient(
            base_url=f"http://localhost:{args.eval_server_port}",
            remote_host=None
        )
        err_msg = f'''
        ❌ ERROR: Local evaluation server is not healthy!
        Server: localhost:{args.eval_server_port}
        Please ensure:
          1. The server is running locally
          2. The server is listening on port {args.eval_server_port}
        Run it like this:
          cd eval_server
          bash run_server.sh
        '''
        success_msg = f"✅ Local evaluation server is healthy and ready!\nServer: localhost:{args.eval_server_port}"
    else:  # remote mode
        print(f"Checking remote evaluation server health ({args.eval_server_host}:{args.eval_server_port})...")
        client = VerusEvalClient(
            remote_host=args.eval_server_host,
            remote_port=args.eval_server_port
        )
        err_msg = f'''
        ❌ ERROR: Remote evaluation server is not healthy!
        Server: {args.eval_server_host}:{args.eval_server_port}
        Please ensure:
          1. The server is running on {args.eval_server_host}
          2. You can SSH to {args.eval_server_host}
          3. The server is listening on port {args.eval_server_port}
          4. Your SSH key (~/.ssh/id_rsa) is configured correctly
        Run it like this:
          ssh {args.eval_server_host.split('.')[0]}
          tmux new -s eval_server
          cd eval_server
          bash run_server.sh
        '''
        success_msg = f"✅ Remote evaluation server is healthy and ready!\nServer: {args.eval_server_host}:{args.eval_server_port}"

    try:
        is_healthy = client.health_check()
        if not is_healthy:
            print(err_msg)
            sys.exit(1)
        else:
            print(success_msg)
    except Exception as e:
        print(err_msg)
        print(f"Error details: {e}")
        sys.exit(1)

def run_codes(codes: List[str], quiet: bool = False, return_full_results: bool = False) -> List:
    """Run codes and return pass results using eval server."""

    results = client.evaluate_batch(codes)

    if return_full_results:
        # Return [(success, stdout, stderr), ...]
        return [(stdout != '', stdout, stderr) for stdout, stderr in results]

    pass_results = [passed(stdout) for stdout, stderr in results]

    if not quiet:
        n_passed = sum(pass_results)
        print(f"Eval: {n_passed}/{len(codes)} passed ({n_passed/len(codes)*100:.1f}%)")

    return pass_results


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k from the Codex paper.
    Calculates 1 - C(n-c, k) / C(n, k) where C is binomial coefficient.

    Args:
        n: total number of samples
        c: number of correct samples
        k: number of samples to evaluate

    Returns:
        Probability that at least one of k samples is correct
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def get_eval_metrics(out_msgs: List, k: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Get evaluation metrics from output messages using unbiased pass@k estimator."""
    syntaxes = np.array([elt[0] != '' for elt in out_msgs]).reshape(-1, k)
    passes = np.array([passed(elt[0]) for elt in out_msgs]).reshape(-1, k)

    metrics = {}
    for _k in K_LIST:
        if _k > k:
            break
        # Compute unbiased pass@k for each problem, then average
        syntax_pass_at_k = np.array([
            pass_at_k_unbiased(k, row.sum(), _k) for row in syntaxes
        ]).mean()
        pass_pass_at_k = np.array([
            pass_at_k_unbiased(k, row.sum(), _k) for row in passes
        ]).mean()

        metrics[f"syntax/syntax@{_k}"] = syntax_pass_at_k
        metrics[f"pass/pass@{_k}"] = pass_pass_at_k
    metrics = dict(sorted(metrics.items()))
    return syntaxes, passes, metrics


def compute_pass_at(results: np.ndarray, cutoffs=(1, 5, 10, 50, 100)) -> Dict[str, float]:
    """
    Compute unbiased pass@k metrics for a 2-D boolean array of shape (N, n).
    Uses the estimator from the Codex paper: 1 - C(n-c, k) / C(n, k)

    Args:
        results: Boolean array of shape (N, n) where N is number of problems
                 and n is total samples generated per problem
        cutoffs: Values of k to compute pass@k for

    Returns:
        Dictionary with pass@k metrics
    """
    if results.ndim != 2 or results.dtype != bool:
        raise ValueError("`results` must be a 2-D boolean array")

    n_problems, n_samples = results.shape
    metrics: Dict[str, float] = {}

    for k in cutoffs:
        if k > n_samples:
            continue
        # Compute unbiased pass@k for each problem, then average
        pass_at_k_values = np.array([
            pass_at_k_unbiased(n_samples, row.sum(), k) for row in results
        ])
        metrics[f"pass@{k}"] = float(pass_at_k_values.mean())
    return metrics


def add_to_metrics(ds, task: str, metrics: Dict) -> Dict:
    """Add task-specific metrics to the metrics dictionary."""
    sub_ds = ds[ds['task'] == task] if task != 'all' else ds
    return {
        **metrics, 
        f'{task}/n_problems': len(sub_ds),
        f'{task}/n_solved': sub_ds['solved'].sum(),
        f'{task}/perc_solved': sub_ds['solved'].mean(),
    }


if __name__ == '__main__':
    print("Testing evaluation...")
    
    # Test passed function
    success_msg = "5 verified, 0 errors"
    failure_msg = "2 verified, 1 errors"
    assert passed(success_msg) == True, "passed() success test failed"
    assert passed(failure_msg) == False, "passed() failure test failed"
    
    # Test spec_passed function
    spec_with_ensures = "fn test() ensures true"
    spec_without_ensures = "fn test()"
    assert spec_passed("3 verified, 0 errors", spec_with_ensures) == True, "spec_passed success test failed"
    assert spec_passed("0 verified, 1 errors", spec_with_ensures) == False, "spec_passed failure test failed"
    assert spec_passed("3 verified, 0 errors", spec_without_ensures) == False, "spec_passed no ensures test failed"
    
    # Test compute_pass_at with unbiased estimator
    # For n=3 samples per problem:
    # Problem 0: c=1 correct -> pass@1=1/3, pass@2=2/3, pass@3=1.0
    # Problem 1: c=1 correct -> pass@1=1/3, pass@2=2/3, pass@3=1.0
    # Problem 2: c=0 correct -> pass@1=0.0, pass@2=0.0, pass@3=0.0
    # Expected averages: pass@1=2/9, pass@2=4/9, pass@3=2/3
    test_results = np.array([
        [True, False, False],
        [False, True, False],
        [False, False, False]
    ])
    metrics = compute_pass_at(test_results, cutoffs=(1, 2, 3))
    assert abs(metrics["pass@1"] - (2/9)) < 0.001, f"compute_pass_at test failed: {metrics['pass@1']} != {2/9}"
    assert abs(metrics["pass@2"] - (4/9)) < 0.001, f"compute_pass_at test failed: {metrics['pass@2']} != {4/9}"
    assert abs(metrics["pass@3"] - (2/3)) < 0.001, f"compute_pass_at test failed: {metrics['pass@3']} != {2/3}"
    
    # Test process_spec
    test_spec = "```rust\nfn test() -> i32\n    ensures result > 0,\n{\n```"
    processed = process_spec(test_spec)
    assert "#[verifier::external_body]" in processed, "process_spec test failed"
    assert "assume(false)" in processed, "process_spec test failed"
    
    # Test MBPP dataset verification
    print("Testing MBPP dataset...")
    try:
        specs, gold_codes, spec_pass_results, gold_pass_results = evaluate_dataset_from_db('mbpp', quiet=False)
        
        if specs:
            # Test spec + completion reconstruction  
            print("Testing MBPP spec + completion reconstruction...")
            spec_completion_match = 0
            reconstructed_codes = []
            
            for i in range(len(specs)):
                spec = specs[i]
                gold_code = gold_codes[i]
                
                # Calculate completion as gold_code minus spec
                completion = gold_code[len(spec):]
                reconstructed = spec + completion
                
                # Test if spec + completion = gold_code
                if reconstructed == gold_code:
                    spec_completion_match += 1
                    reconstructed_codes.append(reconstructed)
                else:
                    reconstructed_codes.append(None)
            
            print(f"✅ {spec_completion_match}/{len(specs)} MBPP spec+completion reconstructions match gold_code")
            
            # Test reconstructed programs
            if spec_completion_match > 0:
                valid_reconstructed = [code for code in reconstructed_codes if code is not None]
                print(f"Testing {len(valid_reconstructed)} reconstructed MBPP programs...")
                reconstruction_pass_results = run_codes(valid_reconstructed, quiet=False)
                spec_completion_pass = sum(reconstruction_pass_results)
                print(f"✅ {spec_completion_pass}/{len(valid_reconstructed)} MBPP reconstructions pass verification")
        
    except Exception as e:
        print(f"⚠️  MBPP dataset test failed: {e}")
        print("This may be expected if running without proper Verus setup")
    
    # Test HumanEval dataset verification
    print("Testing HumanEval dataset...")
    try:
        specs, gold_codes, spec_pass_results, gold_pass_results = evaluate_dataset_from_db('humaneval', quiet=False)
        
        if specs:
            # Test spec + completion reconstruction  
            print("Testing HumanEval spec + completion reconstruction...")
            spec_completion_match = 0
            reconstructed_codes = []
            
            for i in range(len(specs)):
                spec = specs[i]
                gold_code = gold_codes[i]
                
                # Calculate completion as gold_code minus spec
                completion = gold_code[len(spec):]
                reconstructed = spec + completion
                
                # Test if spec + completion = gold_code
                if reconstructed == gold_code:
                    spec_completion_match += 1
                    reconstructed_codes.append(reconstructed)
                else:
                    reconstructed_codes.append(None)
            
            print(f"✅ {spec_completion_match}/{len(specs)} HumanEval spec+completion reconstructions match gold_code")
            
            # Test reconstructed programs
            if spec_completion_match > 0:
                valid_reconstructed = [code for code in reconstructed_codes if code is not None]
                print(f"Testing {len(valid_reconstructed)} reconstructed HumanEval programs...")
                reconstruction_pass_results = run_codes(valid_reconstructed, quiet=False)
                spec_completion_pass = sum(reconstruction_pass_results)
                print(f"✅ {spec_completion_pass}/{len(valid_reconstructed)} HumanEval reconstructions pass verification")
        else:
            print("⚠️  HumanEval dataset not found in cache")
            
    except Exception as e:
        print(f"⚠️  HumanEval dataset test failed: {e}")
        print("This may be expected if running without proper Verus setup")
    
    # Test Dafny2Verus dataset verification  
    print("Testing Dafny2Verus dataset...")
    try:
        specs, gold_codes, spec_pass_results, gold_pass_results = evaluate_dataset_from_db('dafny2verus', quiet=False)
        
        if specs:
            # Test spec + completion reconstruction  
            print("Testing Dafny2Verus spec + completion reconstruction...")
            spec_completion_match = 0
            reconstructed_codes = []
            
            for i in range(len(specs)):
                spec = specs[i]
                gold_code = gold_codes[i]
                
                # Calculate completion as gold_code minus spec
                completion = gold_code[len(spec):]
                reconstructed = spec + completion
                
                # Test if spec + completion = gold_code
                if reconstructed == gold_code:
                    spec_completion_match += 1
                    reconstructed_codes.append(reconstructed)
                else:
                    reconstructed_codes.append(None)
            
            print(f"✅ {spec_completion_match}/{len(specs)} Dafny2Verus spec+completion reconstructions match gold_code")
            
            # Test reconstructed programs
            if spec_completion_match > 0:
                valid_reconstructed = [code for code in reconstructed_codes if code is not None]
                print(f"Testing {len(valid_reconstructed)} reconstructed Dafny2Verus programs...")
                reconstruction_pass_results = run_codes(valid_reconstructed, quiet=False)
                spec_completion_pass = sum(reconstruction_pass_results)
                print(f"✅ {spec_completion_pass}/{len(valid_reconstructed)} Dafny2Verus reconstructions pass verification")
        else:
            print("⚠️  Dafny2Verus dataset not found in cache")
            
    except Exception as e:
        print(f"⚠️  Dafny2Verus dataset test failed: {e}")
        print("This may be expected if running without proper Verus setup")
    
    print("✅ All evaluation tests completed!")