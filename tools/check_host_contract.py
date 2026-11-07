#!/usr/bin/env python3
"""Verify current GDN Host source without importing the NPU runtime."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_hust_gdn_optimizations.contracts import (
    HOST_REPOSITORY,
    HOST_REVISION,
    verify_host,
)

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {sys.argv[0]} /path/to/vllm-ascend-hust")
print(
    json.dumps(
        {
            "repository": HOST_REPOSITORY,
            "revision": HOST_REVISION,
            "status": "pass",
            "files": verify_host(sys.argv[1]),
        },
        indent=2,
        sort_keys=True,
    )
)
