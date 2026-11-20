"""Silver-labeling pipeline for a fine-tune training set (`01-spec/
training-data-2026-09-19.md`, "Next step"). Builds `TrainingRecord`s
(`portolan.training.records`) from the public corpora under
`var/data/converted/`, using the production extraction prompt/schema
(`portolan.extract`) and two free open-weight OpenRouter teachers per
document, keeping only facts both teachers agree on
(`portolan.training.agreement`). See `scripts/data/silver_label.py` for
the CLI that wires this package to real corpora and a real
`OpenRouterClient`.
"""

from __future__ import annotations
