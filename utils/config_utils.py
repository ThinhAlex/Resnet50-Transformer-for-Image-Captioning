"""Configuration helper functions.

This module centralizes YAML loading and simple normalization utilities, so
`train.py` and `inference_save.py` can stay concise and readable.
"""

from __future__ import annotations

import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration and return a copy of the mapping."""
    with open(path, 'r') as f:
        return dict(yaml.safe_load(f) or {})


def _to_number(val: Any, target_type):
    """Convert a config value to the expected numeric type.

    Supports strings, single-item lists, and int/float values.
    """
    if isinstance(val, list) or isinstance(val, tuple):
        if len(val) == 1:
            val = val[0]
        else:
            raise ValueError(f"List provided where numeric expected: {val}")

    if isinstance(val, str):
        try:
            v = eval(val)
        except Exception:
            v = val
        val = v

    return target_type(val)


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `cfg` with typed fields coerced to numeric types.

    This function does not modify the input mapping; it returns a **new** dict.
    """
    out = dict(cfg)
    int_keys = ['batch_size', 'hidden_size', 'num_layers', 'epochs', 'img_size', 'num_workers']
    float_keys = ['learning_rate']

    for k in int_keys:
        if k in out and out[k] is not None:
            out[k] = _to_number(out[k], int)
    for k in float_keys:
        if k in out and out[k] is not None:
            out[k] = _to_number(out[k], float)

    return out
