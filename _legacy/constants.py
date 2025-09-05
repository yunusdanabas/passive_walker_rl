"""
Constants for the legacy passive walker environment.
"""

from pathlib import Path

# Root directory of the project
ROOT = Path(__file__).resolve().parents[1]

# Path to MuJoCo XML model
XML_PATH = ROOT / "passive_walker" / "assets" / "passiveWalker_model.xml"
