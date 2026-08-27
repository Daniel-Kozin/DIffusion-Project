import argparse
from pathlib import Path

from visualize_inserts_3d import visualize_inserts_3d

DATA_DIR = Path(__file__).parent / "mri_images_3D"

parser = argparse.ArgumentParser(description="Interactively view phantoms in a GUI window.")
parser.add_argument("--info", action="store_true",
                     help="Show name + per-phantom data label (no explanatory text).")
parser.add_argument("--help-text", action="store_true",
                     help="Show the full instructions overlay (mouse/key controls).")
args = parser.parse_args()

viz = visualize_inserts_3d(name="view", off_screen=False, show_text=args.help_text, show_info=args.info)
viz.load_folder(DATA_DIR)
viz.show()
