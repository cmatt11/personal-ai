"""Optional encryption-at-rest for conversation content.

When you set a passphrase (PERSONAL_AI_PASSPHRASE) and have the optional
`cryptography` package installed, message and memory text is encrypted before
it is written to the local database. Embeddings are computed from the plaintext
first, so semantic recall still works.

Without a passphrase, or without the package, data is stored as plaintext and
the assistant runs normally. This module never crashes the app.

Note: for full database-file encryption, use SQLCipher. This layer protects the
text content, which is the sensitive part.
"""

import base64
import hashlib
from typing import Optional

from .config import Config

_SALT = b"personal-ai-v1-salt"


class Cipher:
    enabled = False

    def encrypt(self, text: str) -> str:
        return text

    def decrypt(self, text: str) -> str:
        return text


class NullCipher(Cipher):
    """No-op cipher used when encryption is off."""


class FernetCipher(Cipher):
    """Authenticated encryption via the optional `cryptography` package."""

    enabled = True

    def __init__(self, passphrase: str) -> None:
        from cryptography.fernet import Fernet  # imported lazily

        key = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), _SALT, 200_000)
        self._fernet = Fernet(base64.urlsafe_b64encode(key))
        self._prefix = "enc:"

    def encrypt(self, text: str) -> str:
        if text is None:
            return text
        token = self._fernet.encrypt(text.encode("utf-8")).decode("ascii")
        return self._prefix + token

    def decrypt(self, text: str) -> str:
        if not text or not text.startswith(self._prefix):
            return text  # tolerate pre-existing plaintext rows
        try:
            return self._fernet.decrypt(text[len(self._prefix) :].encode("ascii")).decode("utf-8")
        except Exception:
            return text


def cryptography_available() -> bool:
    try:
        import cryptography.fernet  # noqa: F401

        return True
    except Exception:
        return False


def make_cipher(config: Config) -> Cipher:
    """Build a cipher from config, or a NullCipher if not configured/available."""
    passphrase = getattr(config, "passphrase", None)
    if not passphrase:
        return NullCipher()
    if not cryptography_available():
        import sys

        sys.stderr.write(
            "[crypto] PERSONAL_AI_PASSPHRASE is set but the 'cryptography' package "
            "is not installed; storing data as plaintext. Run: pip install cryptography\n"
        )
        return NullCipher()
    return FernetCipher(passphrase)
