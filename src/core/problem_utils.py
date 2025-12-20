"""Problem proposal helper utilities for AV2."""
import re
from typing import List, Set


def parse_nl_resp(resp: str) -> List[str]:
    """Parse natural language response into individual questions."""
    questions = []
    current_question = []
    for line in resp.split('\n'):
        if line.strip():
            first_token = line.strip().split('.')[0]
            if first_token.isdigit() and line.strip().startswith(first_token + '. Write a formally'):
                if current_question:
                    questions.append('\n'.join(current_question))
                current_question = [line[line.index('.')+1:].strip()]
            else:
                current_question.append(line.strip())
    if current_question:
        questions.append('\n'.join(current_question))
    
    return questions


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


def first_unique_indices(function_names: List[str], seen: Set[str]) -> List[int]:
    """Get indices of first occurrence of each unique function name."""
    indices = []
    for idx, name in enumerate(function_names):
        if name not in seen:
            seen.add(name)
            indices.append(idx)
    return indices


if __name__ == '__main__':
    print("Testing problem_utils...")
    
    # Test parse_nl_resp
    test_resp = """1. Write a formally verified function that adds two numbers.
This function should take two integers.

2. Write a formally verified function that finds maximum.
This function should work on arrays."""
    
    questions = parse_nl_resp(test_resp)
    assert len(questions) == 2, f"parse_nl_resp test failed: got {len(questions)} questions"
    assert "adds two numbers" in questions[0], "parse_nl_resp content test failed"
    assert "finds maximum" in questions[1], "parse_nl_resp content test failed"
    
    # Test get_last_fn_name
    rust_code = """
    fn first_function() -> i32 { 42 }
    
    pub fn second_function() -> String { 
        "hello".to_string()
    }
    
    async fn last_function() -> () {}
    """
    last_fn = get_last_fn_name(rust_code)
    assert last_fn == "last_function", f"get_last_fn_name test failed: {last_fn}"
    
    # Test get_last_fn_name with no functions
    empty_code = "let x = 42;"
    no_fn = get_last_fn_name(empty_code)
    assert no_fn is None, f"get_last_fn_name empty test failed: {no_fn}"
    
    # Test first_unique_indices
    function_names = ["add", "sub", "add", "mul", "sub", "div"]
    seen = set()
    indices = first_unique_indices(function_names, seen)
    expected_indices = [0, 1, 3, 5]  # first occurrence of each unique name
    assert indices == expected_indices, f"first_unique_indices test failed: {indices}"
    assert seen == {"add", "sub", "mul", "div"}, f"first_unique_indices seen test failed: {seen}"
    
    # Test with empty list
    empty_indices = first_unique_indices([], set())
    assert empty_indices == [], "first_unique_indices empty test failed"
    
    print("✅ All problem_utils tests passed!")