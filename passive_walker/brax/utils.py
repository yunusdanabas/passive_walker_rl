"""
Utility helpers for the Brax sub-package
========================================

* ``summarize_brax_system`` – pretty HTML / DataFrame dump of a
  :class:`brax.System`.
* ``check_and_save``           – quick numeric sanity check **+** gzip-pickle
  to disk (with an optional MuJoCo joint/actuator dump for debugging).

Both functions are *pure helpers* – they never mutate the Brax ``System`` you
pass in.
"""

from __future__ import annotations
from pathlib import Path
import gzip, pickle, sys as _sys

import numpy as np
import pandas as pd
import jax.numpy as jnp
from IPython.display import Markdown, display


# ---------------------------------------------------------------------------
# Pretty HTML / DataFrame dump
# ---------------------------------------------------------------------------
def summarize_brax_system(sys, *, style: str = "html"):
    """
    Return three :class:`pandas.DataFrame` objects summarising **joints**,
    **actuators** and **bodies** of a Brax ``System``.

    Parameters
    ----------
    sys
        The Brax :class:`brax.System` to visualise.
    style  : {'html', 'df'}
        ``'html'``  – display nice, coloured tables inside a Jupyter notebook  
        ``'df'``    – *do not* display, simply **return** the three dataframes.

    Returns
    -------
    joints_df, actuators_df, bodies_df : pandas.DataFrame
    """
    joint_type_map = {0: "free", 1: "ball", 2: "slide", 3: "hinge", 4: "weld"}
    names_blob = sys.names
    _name = lambda off: names_blob[off:].split(b"\x00", 1)[0].decode()
    body_names = ["world"] + list(sys.link_names)

    tilt = float(jnp.degrees(jnp.arctan2(-sys.gravity[0], sys.gravity[2])))

    # ---------- joints -----------------------------------------------------
    joints_df = pd.DataFrame(
        {
            "idx": np.arange(sys.njnt),
            "type": [joint_type_map.get(int(t), "?") for t in sys.jnt_type],
            "name": [_name(int(o)) for o in sys.name_jntadr],
            "body": [body_names[int(b)] for b in sys.jnt_bodyid],
            "axis": [tuple(a.tolist()) for a in sys.jnt_axis],
            "limited": sys.jnt_limited.astype(bool),
            "lo": sys.jnt_range[:, 0],
            "hi": sys.jnt_range[:, 1],
        }
    )

    # ---------- actuators --------------------------------------------------
    q2joint = {int(qp): idx for idx, qp in enumerate(sys.jnt_qposadr)}
    rows = []
    for i, (lo, hi) in enumerate(sys.actuator.ctrl_range):
        qid, jidx = int(sys.actuator.q_id[i]), q2joint[int(sys.actuator.q_id[i])]
        rows.append(
            dict(
                idx=i,
                name=_name(int(sys.name_actuatoradr[i]))
                if hasattr(sys, "name_actuatoradr")
                else f"act_{i}",
                joint=joints_df.loc[jidx, "name"],
                body=joints_df.loc[jidx, "body"],
                ctrl_lo=lo,
                ctrl_hi=hi,
                unit="rad" if i == 0 else "m",
            )
        )
    actuators_df = pd.DataFrame(rows)

    # ---------- bodies -----------------------------------------------------
    bodies_df = pd.DataFrame(
        {
            "idx": np.arange(len(sys.link_names)),
            "name": sys.link_names,
            "parent": [
                "world" if p == -1 else sys.link_names[p] for p in sys.link_parents
            ],
            "mass": sys.link.inertia.mass,
            "inertia": [tuple(np.diag(I)) for I in sys.link.inertia.i],
        }
    )

    # ---------- optional pretty-print --------------------------------------
    if style == "html":
        display(Markdown("## ⟪ Brax System Summary ⟫"))
        print("‾" * 45)
        print(f" bodies           : {sys.link_names}")
        print(f" nq | nv | nu     : {sys.nq} | {sys.nv} | {sys.nu}")
        print(f" gravity tilt     : {tilt:+.1f} deg")
        display(Markdown("### Bodies"));    display(bodies_df.style.hide(axis="index"))
        display(Markdown("### Joints"));    display(joints_df.style.hide(axis="index"))
        display(Markdown("### Actuators")); display(actuators_df.style.hide(axis="index"))

    return joints_df, actuators_df, bodies_df


# ---------------------------------------------------------------------------
# Sanity-check & pickle helper
# ---------------------------------------------------------------------------
def check_and_save(sys, out: Path, *, verbose: bool = True) -> None:
    """
    * Checks*: finite inertia + positive masses  
    * Side-effect*: gzip-pickle the Brax ``System`` to *out*.

    Parameters
    ----------
    sys   : brax.System
    out   : pathlib.Path
        Destination (``.pkl`` or ``.pkl.gz`` – compression auto-detected).
    verbose : bool
        If *True*, prints per-joint type/name dump **and** a success banner.
    """
    # 1 – verbose dump of MuJoCo joint codes → names
    if verbose:
        print("Raw MuJoCo joint codes ➜ names")
        for idx, (code, name_off) in enumerate(
            zip(sys.jnt_type, sys.name_jntadr, strict=False)
        ):
            name = sys.names[name_off:].split(b"\0", 1)[0]
            print(f"  {idx:2d}  code={int(code)}  name={name.decode()}")

    # 2 – numeric sanity checks
    assert np.all(np.isfinite(sys.link.inertia.i)), "non-finite inertia detected"
    assert np.all(
        sys.link.inertia.mass > 0
    ), "found link(s) with zero or negative mass"

    # 3 – serialise
    out.parent.mkdir(parents=True, exist_ok=True)
    pickler = gzip.open if out.suffix == ".gz" else open
    with pickler(out, "wb") as f:
        pickle.dump(sys, f)

    if verbose:
        print("Basic numeric sanity checks passed ✓")
        print(f"System pickled to: {out.resolve()}")
        print("-" * 40)
