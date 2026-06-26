"""Shared utilities for the self-prediction capabilities research project.

Imported unchanged by every experiment (see EVAL_SPEC_V1.md sec 11). Modules:

- ``types``      shared dataclasses (Dimension, Rollout, Outcome, Forecast, Cell)
- ``llm_client`` OpenRouter client: reasoning-param control, retries, cost accounting
- ``extraction`` two-stage forecast-JSON extractor + gpt-4o-mini fallback
- ``scoring``    Brier / absolute loss, per-dimension standardization, hierarchical bootstrap
- ``stats``      generic inference: bootstrap CIs, mixed model, Spearman, permutation
"""
