"""Prompting and inference preparation utilities for AV2."""
import numpy as np
import json
import os
import random
import re
from os.path import join
from typing import List, Dict
import pandas as pd

# Global cache for random examples
all_random_examples = None


def pull_random_examples(k: int = 5) -> str:
    """Pull k random Verus examples for few-shot prompting."""
    global all_random_examples
    if all_random_examples is None:
        all_random_examples = json.load(open('data/dafny2verus_data_iter6.json'))
        exemplars = []
        for x, y in all_random_examples['solved_pairs']:
            code = open(join('data', y)).read()
            exemplars.append(code)
        all_random_examples = exemplars

    choices = random.sample(all_random_examples, k=k)
    for i, choice in enumerate(choices):
        idx = choice[:choice.find('fn main()')].rfind('fn ') 
        non_body_code = choice[:idx+choice[idx:].find('\n{')+2].strip()
        completion = choice[idx+choice[idx:].find('\n{')+3:].strip()
        return_string += f"## Verus Example {i+1}:\n```rust\n{non_body_code}\n```\n```rust\n{completion}\n```\n\n"
        return_string += '\n'
    return return_string


def get_spec_completion_pairs(ds: pd.DataFrame, use_human_data: bool = True):
    """Get spec-completion pairs for few-shot prompting.

    Args:
        ds: Dataset containing specs and solutions
        use_human_data: Whether to include gold/human data from train datasets

    Returns:
        List of (spec, completion) tuples, deduplicated by spec
    """
    # Filter out test datasets
    ds = ds.loc[~ds.is_test_dataset]

    # Use generated passing solutions always
    spec_completion_pairs = []
    unique_passing_specs = set()

    for _, row in ds.loc[ds['passing_solution'].notna() & (ds['passing_solution'] != '')].iterrows():
        if row['spec'] not in unique_passing_specs:
            spec_completion_pairs.append((row['spec'], row['passing_solution']))
            unique_passing_specs.add(row['spec'])

    # Prepare spec/completion pairs based on use_human_data flag
    if use_human_data:
        # Use gold/human data from train datasets
        for _, row in ds.loc[ds.is_train_dataset].iterrows():
            # Only add if we have gold_code, spec is in gold_code, and spec not already added
            if ('gold_code' in row and
                row['gold_code'] and
                row['spec'] in row['gold_code'] and
                row['spec'] not in unique_passing_specs):
                completion = row['gold_code'][len(row['spec']):]
                spec_completion_pairs.append((row['spec'], completion))
                unique_passing_specs.add(row['spec'])

    return spec_completion_pairs


def get_inf_prompt(spec_in: str, k: int, spec_completion_pairs) -> List[Dict[str, str]]:
    """Generate a prompt for Verus code completion.

    Args:
        spec_in: The input spec to generate completion for
        k: Number of few-shot examples to include
        spec_completion_pairs: List of (spec, completion) tuples to sample from

    Returns:
        List of message dictionaries for the prompt
    """
    msgs = [{'role': 'system', 'content': 'You are a helpful and knowledgeable programming assistant with a strong background in Verus (Rust) and conversant in natural language.'}]
    paran = '{}'
    paran_close = '}'

    # Filter out the current spec from the pairs
    filtered_pairs = [(spec, completion) for spec, completion in spec_completion_pairs if spec not in spec_in]

    if len(filtered_pairs) > 0 and k > 0:
        sampled_pairs = random.sample(filtered_pairs, k=min(k, len(filtered_pairs)))
        for idx, (spec, completion) in enumerate(sampled_pairs):
            msgs.extend([
                { 'role': 'user', 'content': f"Consider the following verus code:\n```rust\n{spec}\n```\n\nThe code contains the relevant spec functions and the preconditions (requires) and postconditions (ensures) for the main function. Your goal is to complete the function, by adding necessary procedure, along with proof statements (such as invariants, asserts, proof blocks etc) to prove the program. Only output the new program and not the entire code. You are not allowed to create new functions, however can use any functions already defined if within the scope. Remember to just output the completion without the function signature, requires and ensures. Only the body of the function is required. Remember to end in: \n```rust\n{paran_close} // End of function\nfn main() {paran}\n{paran_close} // verus!\n\n```\n\n" },
                { 'role': 'assistant', 'content': f"```rust\n{completion}\n```" }
            ])
    msgs.append( { 'role': 'user', 'content': f"Consider the following verus code:\n```rust\n{spec_in}\n```\n\nThe code contains the relevant spec functions and the preconditions (requires) and postconditions (ensures) for the main function. Your goal is to complete the function, by adding necessary procedure, along with proof statements (such as invariants, asserts, proof blocks etc) to prove the program. Only output the new program and not the entire code. You are not allowed to create new functions, however can use any functions already defined if within the scope. Remember to just output the completion without the function signature, requires and ensures. Only the body of the function is required. Remember to end in: \n```rust\n{paran_close} // End of function\nfn main() {paran}\n{paran_close} // verus!\n\n```\n\n" })
    return msgs

def fix_verus_code_full(src: str) -> str:
    """
    Steps:
      1) Remove any 'fn main () {}' or 'fn main() {}' lines (spaces flexible).
      2) Balance curly braces by removing extra '}' from the back.
      3) Insert 'fn main () {}' on its own line just before the final '}'.
    """
    # 1) Remove existing mains (with flexible spacing and optional trailing comment)
    main_line_re = re.compile(
        r'(?m)^[ \t]*fn\s+main\s*\(\s*\)\s*\{\s*\}\s*(?:\/\/[^\n]*)?\s*\n?'
    )
    s = main_line_re.sub('', src)

    # 2) Balance braces by trimming extra '}' from the end
    left = s.count('{')
    right = s.count('}')
    to_trim = right - left
    while to_trim > 0:
        idx = s.rfind('}')
        if idx == -1:
            break
        s = s[:idx] + s[idx+1:]
        to_trim -= 1

    # 3) Insert 'fn main () {}' just before the final '}'
    insert_line = 'fn main () {}\n'
    last_close = s.rfind('}')
    if last_close != -1:
        prefix, suffix = s[:last_close], s[last_close:]
        if prefix and not prefix.endswith('\n'):
            prefix += '\n'
        return prefix + insert_line + suffix
    else:
        # No closing brace found; just append on a new line
        if not s.endswith('\n'):
            s += '\n'
        return s + insert_line


# Rust code block extraction
RUST_BLOCK_RE = re.compile(
    r"```rust\s*([\s\S]*?)```",   # non-greedy capture of everything up to the next ```
    re.IGNORECASE                 # make "rust" case-insensitive just in case
)


def extract_rust_block(text: str) -> str:
    """Return the first Rust fenced-code block found in text, or empty string if none."""
    try:
        m = RUST_BLOCK_RE.search(text)
    except Exception as e:
        print(f"extract_rust_block: error searching text: {e}")
        return ''
    return m.group(1).strip() if m else ''


def get_last_fn_name(rust_source: str) -> str | None:
    """Extract the last function name from Rust source code."""
    rx = re.compile(r"""
        ^\s*                                   # line start + leading spaces
        (?:pub\s*(?:\([^)]+\))?\s*)?           # optional pub / pub(...)
        (?:async\s+)?                          # optional async
        (?:const\s+)?                          # optional const
        (?:unsafe\s+)?                         # optional unsafe
        (?:extern\s*(?:"[^"]*"\s*)?)?          # optional extern "ABI"
        fn\s+                                  # the fn keyword
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)       # ← capture the function name
        (?:\s*<[^>]*>\s*)?                     # ← optional generics after the name, e.g. <'a, T, const N: usize>
        \s*\(                                  # up to the first '(' of the parameter list
    """, re.MULTILINE | re.VERBOSE)
    
    matches = [m.group('name') for m in rx.finditer(rust_source)]
    return matches[-1] if matches else None


def postprocess_verus(spec: str, completion: str) -> str:
    """
    Returns the cleaned completion per the rules described. 
    Combines spec and completion and ensures proper structure.
    """
    completion = extract_rust_block(completion)
    
    if completion == '':
        return ''
    
    last_fn_name = get_last_fn_name(spec)
    if last_fn_name is None:
        return ''
    spec_fn_onward = spec[spec.index(last_fn_name):]
    if spec_fn_onward in completion:  # full solution
        completion = completion.split(spec_fn_onward)[1]

    # add spec, make sure curly braces are balanced with spec, add fn main() {}
    full_code = spec + '\n' + completion
    full_code_fixed = fix_verus_code_full(full_code)

    # then grab only spec onwards
    if spec not in full_code_fixed:
        return ''
    split = full_code_fixed.split(spec)
    if len(split) == 1:
        return ''
    completion = split[1]
    return completion


if __name__ == '__main__':
    print("Testing prompt_utils...")
    
    # Test extract_rust_block
    test_text = "Some text\n```rust\nfn test() { }\n```\nMore text"
    result = extract_rust_block(test_text)
    expected = "fn test() { }"
    assert result == expected, f"extract_rust_block test failed: {result}"
    
    # Test empty rust block
    empty_text = "No rust code here"
    empty_result = extract_rust_block(empty_text)
    assert empty_result == '', f"Empty rust block test failed: {empty_result}"
    
    # Test get_last_fn_name
    rust_code = """
    fn first() {}
    fn second() {}
    """
    last_fn = get_last_fn_name(rust_code)
    assert last_fn == "second", f"get_last_fn_name test failed: {last_fn}"
    
    # Test fix_verus_code_full basic functionality
    test_code = "fn test() {\n    return 42;\n}\nfn main() {}"
    fixed_code = fix_verus_code_full(test_code)
    assert "fn main ()" in fixed_code, f"fix_verus_code_full test failed: {fixed_code}"
    
    # Test get_prompt structure
    spec = "fn test() -> i32"
    prompt = get_prompt(spec, 0)  # k=0 to avoid file dependencies
    assert len(prompt) == 2, f"get_prompt structure test failed: {len(prompt)}"
    assert prompt[0]['role'] == 'system', "get_prompt system role test failed"
    assert prompt[1]['role'] == 'user', "get_prompt user role test failed"
    assert spec in prompt[1]['content'], "get_prompt content test failed"
    
    print("✅ All prompt_utils tests passed!")