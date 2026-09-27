"""Portable text-manifest identity, including legacy platform-specific hashes."""
import hashlib
from pathlib import Path


def manifest_sha256(path):
    """New manifests use LF for hashing, independently of checkout platform."""
    data = Path(path).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def manifest_matches(path, expected):
    """Accept only line-ending differences, without rewriting a frozen recipe.

    Old recipes hashed raw bytes on Windows or Linux. Preserve those identities
    for resume while still rejecting any other change, including pool order.
    """
    path = Path(path)
    if not path.is_file() or not isinstance(expected, str):
        return False
    raw = path.read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    return any(hashlib.sha256(data).hexdigest() == expected
               for data in (raw, lf, lf.replace(b'\n', b'\r\n')))
