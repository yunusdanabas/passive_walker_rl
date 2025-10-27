from __future__ import annotations
from pathlib import Path
import warnings
from .paths import (
    EXPERIMENTS_ROOT, PLOTS_DIR, METRICS_DIR, REPORTS_DIR,
    BC_PLOTS_DIR, PPO_PLOTS_DIR
)


LEGACY_TO_NEW = [
    ("outputs/plots", PLOTS_DIR),
    ("evaluation_plots", PLOTS_DIR),
    ("ppo_plots", PPO_PLOTS_DIR),
    ("results/figures", PLOTS_DIR),
    ("evaluation", METRICS_DIR),
    ("evaluation_results", METRICS_DIR),
    ("outputs/metrics", METRICS_DIR),
    ("results/data", METRICS_DIR),
]


def redirect_legacy_dir(path: str | Path) -> Path:
    """Redirect legacy experiment subpaths to the new unified structure.

    Emits a DeprecationWarning when a legacy segment is detected.
    """
    p = Path(path)
    s = str(p)
    for legacy, new_base in LEGACY_TO_NEW:
        if legacy in s:
            warnings.warn(
                f"Deprecated path segment '{legacy}' detected. Redirecting to '{new_base}'.",
                DeprecationWarning,
                stacklevel=2,
            )
            idx = s.index(legacy)
            # Replace the legacy segment with the relative path of the new base under experiments
            replaced = s[idx:].replace(legacy, str(new_base.relative_to(EXPERIMENTS_ROOT)))
            return EXPERIMENTS_ROOT / replaced
    return p


