from pathlib import Path

from visualize_inserts_3d import visualize_inserts_3d

DATA_DIR = Path(__file__).parent / "mri_images_3D"

viz = visualize_inserts_3d(name="showcase", off_screen=True, show_text=False, show_info=False)
viz.load_folder(DATA_DIR)
viz.show()
