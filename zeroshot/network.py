"""Bounded HTTP access, with separate rules for research URLs and chosen LLM endpoints."""
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests


class NetworkError(ValueError):
    pass


def validate_url(url, allow_local=False):
    p = urlparse(url)
    if p.scheme not in {"https", "http"} or not p.hostname or p.username or p.password or p.fragment:
        raise NetworkError("Use an HTTP(S) URL without embedded credentials or a fragment.")
    if allow_local and p.hostname in {"localhost", "127.0.0.1", "::1"}:
        return url
    if p.scheme != "https":
        raise NetworkError("Remote endpoints must use HTTPS.")
    try:
        addresses = socket.getaddrinfo(p.hostname, p.port or 443, type=socket.SOCK_STREAM)
    except OSError:
        raise NetworkError("Could not resolve the remote hostname.") from None
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise NetworkError("Research URLs and remote providers must resolve to public addresses.")
    return url


def get_bytes(url, limit=8_000_000):
    """Validate every redirect; never forward provider credentials to paper sources."""
    for _ in range(6):
        validate_url(url)
        try:
            with requests.get(url, timeout=(10, 35), allow_redirects=False, stream=True,
                              headers={"User-Agent": "ZeroShot/0.1 (research evidence retrieval)"}) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get("Location", ""))
                    continue
                if not response.ok:
                    raise NetworkError(f"Source returned HTTP {response.status_code}.")
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > limit:
                        raise NetworkError("Source exceeds the 8 MB retrieval limit.")
                    chunks.append(chunk)
                return b"".join(chunks), response.headers.get("Content-Type", ""), url
        except requests.RequestException:
            raise NetworkError("Source request failed or timed out.") from None
    raise NetworkError("Source redirected too many times.")


def post_json(url, headers, payload, allow_local=False):
    validate_url(url, allow_local=allow_local)
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=(10, 120),
                                 allow_redirects=False)
        if not response.ok or response.is_redirect:
            # Do not echo a provider's body: it could contain credentials or private prompts.
            raise NetworkError(f"LLM returned HTTP {response.status_code}. Check model, endpoint, token and quota.")
        return response.json()
    except (requests.RequestException, requests.exceptions.JSONDecodeError):
        raise NetworkError("LLM request failed, timed out, or returned invalid JSON.") from None

