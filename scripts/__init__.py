"""Package init: make TLS work for the stale-stack tools.

The system Python on this box has no usable CA bundle, so anything that reaches the network
through stdlib `urllib` (easyocr model download, some torch hub paths) dies with
CERTIFICATE_VERIFY_FAILED. Point it at certifi's bundle before any such import. See
build-log.md 2026-09-04.
"""
import os

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:  # noqa: BLE001
    pass
