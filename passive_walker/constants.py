# passive_walker/constants.py
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
XML_PATH   = ROOT / "passiveWalker_model.xml"
BC_DATA    = ROOT / "data" / "bc"
BC_DATA.mkdir(parents=True, exist_ok=True)
