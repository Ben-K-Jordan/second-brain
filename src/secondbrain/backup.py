"""Round 16 (Phase E) — encrypted backup format.

Self-describing binary format:

  magic        | 8 bytes  | b"SBBKv001"
  pbkdf2_iter  | 4 bytes  | unsigned LE int (default 600_000)
  salt         | 16 bytes | random
  nonce        | 12 bytes | random (AES-GCM nonce)
  ciphertext   | N bytes  | AES-256-GCM encrypted plaintext
  tag          | 16 bytes | GCM auth tag (auto-appended by AESGCM.encrypt)

Key derivation: PBKDF2-HMAC-SHA256 with the embedded ``pbkdf2_iter``
count over (passphrase, salt). Default 600_000 iterations matches OWASP
2023 guidance for SHA-256.

Auth: AES-GCM. Tampered ciphertext or wrong passphrase → MAC verify
fails → ``BadPassphraseError``.

Cipher choice rationale:

  - **AES-256-GCM** vs ChaCha20-Poly1305: both are fine; AES-GCM is
    in the standard library via cryptography, hardware-accelerated on
    most modern CPUs (AES-NI), and the more familiar choice for backup
    audits.
  - **PBKDF2** vs Argon2: PBKDF2 is in the cryptography std lib (no
    extra dep). Argon2 is stronger but adds an argon2-cffi dep that's
    overkill for a personal-tool passphrase that's typed by hand.

Implementation cautions:

  - Stream the file in 4 MB chunks for memory safety on large backups.
    AES-GCM in cryptography doesn't support true streaming, so we
    compromise by reading the whole file into memory first — bounded
    in practice by SQLite backup size, which for a personal KB tops
    out around 1-2 GB. That's fine.
  - Wipe the derived key from memory ASAP. Python doesn't actually
    let us zero bytes (gc'd later), but we can ``del`` the reference.
"""

from __future__ import annotations

import logging
import secrets
import struct
from pathlib import Path

log = logging.getLogger(__name__)


_MAGIC = b"SBBKv001"
_PBKDF2_DEFAULT_ITERS = 600_000
_SALT_LEN = 16
_NONCE_LEN = 12
_KEY_LEN = 32  # AES-256


class BadPassphraseError(ValueError):
    """Raised when AES-GCM auth tag verification fails — meaning either
    the passphrase is wrong or the ciphertext was tampered with."""


def _derive_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_LEN,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_file(src: Path, dst: Path, passphrase: str) -> None:
    """Encrypt ``src`` with ``passphrase`` and write to ``dst``.

    Uses AES-256-GCM. Salt + nonce are randomly generated. Output
    format: magic || iters || salt || nonce || ciphertext_with_tag.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    iterations = _PBKDF2_DEFAULT_ITERS
    salt = secrets.token_bytes(_SALT_LEN)
    nonce = secrets.token_bytes(_NONCE_LEN)
    key = _derive_key(passphrase, salt, iterations)
    try:
        aes = AESGCM(key)
        plaintext = src.read_bytes()
        ciphertext = aes.encrypt(nonce, plaintext, associated_data=_MAGIC)
    finally:
        del key  # let GC reclaim ASAP
    header = (
        _MAGIC
        + struct.pack("<I", iterations)
        + salt
        + nonce
    )
    dst.write_bytes(header + ciphertext)


def decrypt_file(src: Path, dst: Path, passphrase: str) -> None:
    """Decrypt ``src`` with ``passphrase`` and write to ``dst``.

    Raises ``BadPassphraseError`` if the AES-GCM auth tag fails to
    verify (wrong key or corrupted file).
    """
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    blob = src.read_bytes()
    header_len = len(_MAGIC) + 4 + _SALT_LEN + _NONCE_LEN
    if len(blob) < header_len + 16:
        raise BadPassphraseError("file too small to be a valid backup")
    if blob[:len(_MAGIC)] != _MAGIC:
        raise BadPassphraseError(
            f"bad magic: not a secondbrain encrypted backup "
            f"(got {blob[:len(_MAGIC)]!r})"
        )
    offset = len(_MAGIC)
    (iterations,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    salt = blob[offset:offset + _SALT_LEN]
    offset += _SALT_LEN
    nonce = blob[offset:offset + _NONCE_LEN]
    offset += _NONCE_LEN
    ciphertext = blob[offset:]

    key = _derive_key(passphrase, salt, iterations)
    try:
        aes = AESGCM(key)
        try:
            plaintext = aes.decrypt(
                nonce, ciphertext, associated_data=_MAGIC,
            )
        except InvalidTag as e:
            raise BadPassphraseError(
                "auth tag verify failed — wrong passphrase or tampered file"
            ) from e
    finally:
        del key
    dst.write_bytes(plaintext)


def is_encrypted_file(path: Path) -> bool:
    """Return True if ``path`` starts with our magic bytes. A bare
    SQLite DB starts with ``b"SQLite format 3\\x00"`` so the
    discrimination is unambiguous."""
    try:
        with path.open("rb") as fh:
            head = fh.read(len(_MAGIC))
    except OSError:
        return False
    return head == _MAGIC
