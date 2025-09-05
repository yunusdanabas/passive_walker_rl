"""
CLI utility – convert MuJoCo XML ➜ Brax System
----------------------------------------------

$ python -m passive_walker.brax.convert_xml --xml passiveWalker_model.xml
"""

from __future__ import annotations
import argparse, pathlib, mujoco
from brax.io import mjcf
from passive_walker.brax import XML_PATH, SYSTEM_PICKLE, set_device
from passive_walker.brax.utils import summarize_brax_system, check_and_save


def _fix_link_types(sys):
    """Tiny patch for models exported by older MuJoCo versions."""
    if len(sys.link_types) == len(sys.link_names):       # nothing to fix
        return sys
    if len(sys.link_types) == len(sys.link_names) - 1:   # prepend world link
        return sys.replace(link_types=sys.link_types[:1] + sys.link_types)
    raise RuntimeError("Unexpected link_types / link_names mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=pathlib.Path, default=XML_PATH,
                        help="MuJoCo XML file (default: passiveWalker_model.xml)")
    parser.add_argument("--out", type=pathlib.Path, default=SYSTEM_PICKLE,
                        help="Pickle destination (default: data/brax/system.pkl.gz)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force JAX-CPU for the conversion step")
    args = parser.parse_args()

    # choose backend *before* importing JAX
    set_device(not args.cpu)

    # 1 – load & convert
    mj_model   = mujoco.MjModel.from_xml_path(str(args.xml))
    brax_sys   = mjcf.load_model(mj_model)
    brax_sys   = _fix_link_types(brax_sys)

    # 2 – pretty print
    summarize_brax_system(brax_sys)

    # 3 – numeric check + pickle
    check_and_save(brax_sys, pathlib.Path(args.out))


if __name__ == "__main__":
    main()
