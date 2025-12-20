"""
Client library for interacting with the Verus evaluation server.

Can be imported and used from other Python scripts.
"""

import requests
import asyncio
import asyncssh
import httpx
import os
from typing import List, Tuple

SSH_KEY = os.path.expanduser('~/.ssh/id_rsa')


class VerusEvalClient:
    """Client for the Verus evaluation server."""

    def __init__(self, base_url: str = "http://localhost:5000", remote_host: str = None, remote_port: int = 5000):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the evaluation server (for local connections)
            remote_host: Remote SSH host for tunneling (optional)
            remote_port: Remote port where the eval server is running (default: 5000)
        """
        self.base_url = base_url.rstrip('/')
        self.remote_host = remote_host
        self.remote_port = remote_port
        self._tunnel = None
        self._local_port = None

    async def _establish_tunnel(self):
        """Establish SSH tunnel to remote server."""
        if self.remote_host is None:
            return

        self._conn = await asyncssh.connect(
            self.remote_host,
            username=os.environ.get('SSH_USER', os.getlogin()),
            port=22,
            client_keys=[SSH_KEY],
            known_hosts=None,
        )
        self._listener = await self._conn.forward_local_port(
            '127.0.0.1', 0, '127.0.0.1', self.remote_port
        )
        self._local_port = self._listener.get_port()

    async def _close_tunnel(self):
        """Close SSH tunnel."""
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None
            self._listener = None
            self._local_port = None

    def _get_base_url(self) -> str:
        """Get the base URL to use (local tunnel or direct)."""
        if self.remote_host and self._local_port:
            return f"http://localhost:{self._local_port}"
        return self.base_url

    def health_check(self) -> bool:
        """
        Check if the server is running and healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        async def _async_health_check():
            await self._establish_tunnel()
            try:
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.get(f"{self._get_base_url()}/health", timeout=10.0)
                    return response.status_code == 200
            except Exception:
                return False
            finally:
                await self._close_tunnel()

        if self.remote_host:
            return asyncio.run(_async_health_check())
        else:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                return response.status_code == 200
            except requests.RequestException:
                return False

    def evaluate(self, code: str, timeout: int = 10) -> Tuple[str, str]:
        """
        Evaluate a single code snippet.

        Args:
            code: Verus code to evaluate
            timeout: Timeout in seconds

        Returns:
            Tuple of (stdout, stderr)
        """
        async def _async_evaluate():
            await self._establish_tunnel()
            try:
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.post(
                        f"{self._get_base_url()}/evaluate",
                        json={"code": code, "timeout": timeout},
                        timeout=timeout + 10  # Add buffer for HTTP timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data.get('stdout', ''), data.get('stderr', '')
            except Exception as e:
                return '', f"Request failed: {str(e)}"
            finally:
                await self._close_tunnel()

        if self.remote_host:
            return asyncio.run(_async_evaluate())
        else:
            try:
                response = requests.post(
                    f"{self.base_url}/evaluate",
                    json={"code": code, "timeout": timeout},
                    timeout=timeout + 5  # Add buffer for HTTP timeout
                )
                response.raise_for_status()
                data = response.json()
                return data.get('stdout', ''), data.get('stderr', '')
            except requests.RequestException as e:
                return '', f"Request failed: {str(e)}"

    def evaluate_batch(self, codes: List[str], timeout: int = 10) -> List[Tuple[str, str]]:
        """
        Evaluate multiple code snippets in a batch.

        Args:
            codes: List of Verus code strings to evaluate
            timeout: Timeout in seconds per code

        Returns:
            List of (stdout, stderr) tuples
        """
        async def _async_evaluate_batch():
            await self._establish_tunnel()
            try:
                # Calculate total timeout based on number of codes
                total_timeout = len(codes) * timeout + 30

                async with httpx.AsyncClient() as http_client:
                    response = await http_client.post(
                        f"{self._get_base_url()}/evaluate_batch",
                        json={"codes": codes, "timeout": timeout},
                        timeout=total_timeout
                    )
                    response.raise_for_status()
                    data = response.json()

                    results = []
                    for result in data.get('results', []):
                        stdout = result.get('stdout', '')
                        stderr = result.get('stderr', '')
                        results.append((stdout, stderr))

                    return results

            except Exception as e:
                # Return error for all codes
                error_msg = f"Batch request failed: {str(e)}"
                return [('', error_msg) for _ in codes]
            finally:
                await self._close_tunnel()

        if self.remote_host:
            return asyncio.run(_async_evaluate_batch())
        else:
            try:
                # Calculate total timeout based on number of codes
                total_timeout = len(codes) * timeout + 30

                response = requests.post(
                    f"{self.base_url}/evaluate_batch",
                    json={"codes": codes, "timeout": timeout},
                    timeout=total_timeout
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for result in data.get('results', []):
                    stdout = result.get('stdout', '')
                    stderr = result.get('stderr', '')
                    results.append((stdout, stderr))

                return results

            except requests.RequestException as e:
                # Return error for all codes
                error_msg = f"Batch request failed: {str(e)}"
                return [('', error_msg) for _ in codes]


def create_client(base_url: str = "http://localhost:5000", remote_host: str = None, remote_port: int = 5000) -> VerusEvalClient:
    """
    Create a VerusEvalClient instance.

    Args:
        base_url: Base URL of the evaluation server (for local connections)
        remote_host: Remote SSH host for tunneling (optional)
        remote_port: Remote port where the eval server is running (default: 5000)

    Returns:
        VerusEvalClient instance
    """
    return VerusEvalClient(base_url, remote_host, remote_port)
