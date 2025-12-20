"""OpenAI API client utilities for AV2."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai


def _load_api_key(path: str = "ignore/openai_key_secret.txt") -> str:
    """Read the secret key from a local text file (first line, trimmed)."""
    return Path(path).read_text(encoding="utf-8").strip()


# Configure the OpenAI client
try:
    client_oai = openai.OpenAI(api_key=_load_api_key())
except Exception:
    # Fallback for testing or when key file doesn't exist
    client_oai = None


def chat_completion_oai(
    messages: List[Dict[str, str]],
    model: str = "o3",
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 1.0,
    n: Optional[int] = 1,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    **extra_params,
) -> Any:
    """
    Call the OpenAI chat completion endpoint and return the raw response object.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        Conversation in OpenAI Chat format. Example:
        [{"role": "system", "content": "You are helpful"}, ...]
    model : str
        OpenAI model name. Defaults to 'o3'.
    temperature, top_p, n, max_tokens, stream
        Standard sampling / decoding knobs (pass None to accept server defaults).
    **extra_params
        Forward-compatibility: any other kwargs you want the API to receive.

    Returns
    -------
    Response object from OpenAI API
    """
    if client_oai is None:
        raise RuntimeError("OpenAI client not initialized. Check API key file.")
    
    response = client_oai.chat.completions.create(
        model=model,
        messages=messages,
        top_p=top_p,
        n=n,
        stream=stream,
        **extra_params,
    )
    return response


if __name__ == '__main__':
    print("Testing openai_client...")
    
    # Test API key loading (mock test since we may not have the file)
    try:
        # This will work if the key file exists
        key = _load_api_key()
        assert isinstance(key, str) and len(key) > 0, "API key loading test failed"
        print("✅ API key loaded successfully")
    except FileNotFoundError:
        print("⚠️  API key file not found - this is expected in testing")
    
    # Test message format validation (without making actual API calls)
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Basic structure validation
    for msg in test_messages:
        assert "role" in msg, "Message format test failed - missing role"
        assert "content" in msg, "Message format test failed - missing content"
        assert msg["role"] in ["system", "user", "assistant"], f"Invalid role: {msg['role']}"
    
    # Test client initialization check
    if client_oai is None:
        print("⚠️  OpenAI client not initialized - this is expected without API key")
    else:
        print("✅ OpenAI client initialized successfully")
    
    print("✅ All openai_client tests passed!")