"""File I/O utilities for AV2."""
import json
import pickle
from typing import Any, List, Dict, Optional


def load_pk(path: str) -> Any:
    """Load a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pk(path: str, obj: Any) -> None:
    """Save an object to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_json(path: str) -> Dict:
    """Load a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: str, data: Dict) -> None:
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f)


def load_txt(path: str, encoding: str = "utf-8") -> str:
    """Load a text file."""
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


def save_txt(path: str, data: str) -> None:
    """Save data to text file."""
    with open(path, 'w') as f:
        f.write(data)


def txt_debug(data: str, path: str = 'hi.txt') -> None:
    """Debug helper to save text to file."""
    save_txt(path, data)


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: str, data: List[Dict]) -> None:
    """Save data to JSONL file."""
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    print("Testing io_utils...")
    
    # Test pickle operations
    test_data = {"test": "value", "number": 42}
    save_pk("/tmp/test.pkl", test_data)
    loaded_data = load_pk("/tmp/test.pkl")
    assert loaded_data == test_data, f"Pickle test failed: {loaded_data}"
    
    # Test JSON operations
    save_json("/tmp/test.json", test_data)
    loaded_json = load_json("/tmp/test.json")
    assert loaded_json == test_data, f"JSON test failed: {loaded_json}"
    
    # Test text operations
    test_text = "Hello, World!\nThis is a test."
    save_txt("/tmp/test.txt", test_text)
    loaded_text = load_txt("/tmp/test.txt")
    assert loaded_text == test_text, f"Text test failed: {loaded_text}"
    
    # Test JSONL operations
    test_jsonl = [{"id": 1, "text": "first"}, {"id": 2, "text": "second"}]
    save_jsonl("/tmp/test.jsonl", test_jsonl)
    loaded_jsonl = load_jsonl("/tmp/test.jsonl")
    assert loaded_jsonl == test_jsonl, f"JSONL test failed: {loaded_jsonl}"
    
    # Clean up
    import os
    for fname in ["/tmp/test.pkl", "/tmp/test.json", "/tmp/test.txt", "/tmp/test.jsonl"]:
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass
    
    print("✅ All io_utils tests passed!")


def generate_model_name_for_cache(
    base_model: str,
    iteration: int,
    proposal_strategy: Optional[str] = None,
    max_n_qs: Optional[int] = None,
    train_dataset: str = "mbpp",
    epochs: Optional[int] = None,
    trained_proposer: bool = False,
    inf_fs: Optional[int] = None,
    inf_k: Optional[int] = None
) -> str:
    """Generate consistent model names for caching.

    Args:
        base_model: Base model name (e.g., 'Qwen/Qwen2.5-Coder-3B-Instruct')
        iteration: Training iteration number (0 = base model, 1+ = LoRA iterations)
        proposal_strategy: Proposal strategy ('e2h', 'h2e', etc.)
        max_n_qs: Maximum number of questions
        train_dataset: Training dataset name
        epochs: Number of training epochs (included in name for iter 1+)
        trained_proposer: Whether using trained proposer
        inf_fs: Number of few-shot examples used in inference
        inf_k: Number of varied prompts per question in inference

    Returns:
        Consistent model name for caching
    """
    base_name = base_model.replace('/', '_')
    dataset_name = train_dataset.upper()

    # Handle None epochs by defaulting to 2
    if epochs is None:
        epochs = 2
        print(f"WARNING: epochs was None, defaulting to {epochs}")

    # Determine data type suffix from proposal_strategy
    data_type_suffix = ""
    if proposal_strategy:
        if "zerodata" in proposal_strategy:
            data_type_suffix = "_zerodata"
        elif "humandata" in proposal_strategy:
            data_type_suffix = "_humandata"
        else:
            data_type_suffix = "_other_data"

    # Add inf_k and inf_fs suffixes if provided
    inf_suffix = ""
    if inf_k is not None:
        inf_suffix += f"_inf_k_{inf_k}"
    if inf_fs is not None:
        inf_suffix += f"_inf_fs_{inf_fs}"

    if "solutionver" in proposal_strategy:
        return f"{base_name}_iter{iteration}_proposal_strat_{proposal_strategy}_base_ds_{dataset_name}_epochs_{epochs}{data_type_suffix}{inf_suffix}_solutionver"

    if iteration == 0:
        # Base model - no LoRA
        return f"{base_name}{data_type_suffix}{inf_suffix}"
    elif iteration == 1:
        # First LoRA iteration - include epochs in name
        return f"{base_name}_iter1_base_ds_{dataset_name}_epochs_{epochs}{data_type_suffix}{inf_suffix}"
    else:
        if iteration >= 3 and trained_proposer:
            return f"{base_name}_iter{iteration}_proposal_strat_{proposal_strategy}_max_n_qs_{max_n_qs}_base_ds_{dataset_name}_epochs_{epochs}_trained_proposer_{trained_proposer}{data_type_suffix}{inf_suffix}"
        if 'rft' in proposal_strategy:
            return f"{base_name}_iter{iteration}_proposal_strat_{proposal_strategy}_base_ds_{dataset_name}_epochs_{epochs}{data_type_suffix}{inf_suffix}"
        # Subsequent iterations with full naming including epochs
        return f"{base_name}_iter{iteration}_proposal_strat_{proposal_strategy}_max_n_qs_{max_n_qs}_base_ds_{dataset_name}_epochs_{epochs}{data_type_suffix}{inf_suffix}"
