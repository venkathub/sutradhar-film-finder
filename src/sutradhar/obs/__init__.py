"""Observability package (P3 task 10): Langfuse tracing + MLflow experiment tracking.

Both are **no-op-safe**: tracing silently disables when ``LANGFUSE_*`` keys are unset
(Tier-1 CI, forks), and MLflow logging degrades with a clear message when the tracking
server is unreachable — evals never fail because observability is off (DEC-P0-4 posture).
"""
