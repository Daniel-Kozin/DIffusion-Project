import os
from pathlib import Path
from typing import Tuple, Sequence, Optional

import pyvista as pv
import numpy as np
import torch

# Type aliases for readability
Vec3 = Tuple[float, float, float]
CameraPosition = Sequence[Vec3]


class visualize_inserts_3d:

    def __init__(self, window_size: (int, int) = (900, 900), off_screen: bool = True, name: str = "",
                 output_path: str = "screenshots", debug: bool = False, show_text: bool = False,
                 show_info: bool = False, look_up = False, should_smooth_mesh = True):

        self.off_screen = off_screen

        self.plotter = pv.Plotter(window_size=window_size, off_screen=self.off_screen)
        self.plotter.set_background('paraviewbackground')
        # removes an error that doesn't affect us
        self.plotter.left_button_down = lambda *args, **kwargs: None
        self.should_smooth_mesh = should_smooth_mesh

        self.gray_actor = None
        self.original_gray_opacity = 0.3
        self.current_lump_opacity = 0.95
        self.current_pillar_opacity = 0.95
        self.current_gray_opacity = self.original_gray_opacity
        self.history = []  # keep track of arrays
        self.index = -1  # current array index
        self.name = name
        self.screenshot_counter = 0
        self.start_flag = False

        self.big_inserts = ["bst12d30", "bst12d15"]
        self.is_big = []
        # Setup output directory
        self.output_path = Path(output_path)
        self.output_dir = self.output_path / self.name
        self.showcase_path = self.output_dir / "showcase"
        self.html_path = self.output_dir / "HTML"

        self.debug = debug
        self.names = []  # keep track of names

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.showcase_path, exist_ok=True)
        os.makedirs(self.html_path, exist_ok=True)

        # Default camera pos
        self.default_cpos = [(100.12834218186312, 150, 21.032991409301758),
                  (2.6999998092651367, 31.271705389022827, 31.032991409301758),
                  (1, 0, 0)]

        # cpos to look up
        if look_up:
            self.default_cpos = [(150.12834218186312, 31.271705389022827, 31.032991409301758),
                (2.6999998092651367, 31.271705389022827, 31.032991409301758),
                (0, -1, 0)]


        # for showcase
        self.showcased = set()  # track which indices already got a showcase screenshot
        self.showcase_cpos = self.default_cpos

        # Key bindings
        self.plotter.add_key_event("q", self.quit_program)
        self.plotter.add_key_event("g", self.toggle_gray)
        self.plotter.add_key_event("n", self.next_array)
        self.plotter.add_key_event("b", self.prev_array)
        self.plotter.add_key_event("r", self.reset_camera)
        self.plotter.add_key_event("s", self.save_screenshot)
        self.plotter.add_key_event("p", self.print_camera)
        self.plotter.add_key_event("h", self.export_to_html)

        self.show_text = show_text
        self.show_info = show_info

    def get_name(self):
        return self.name

    def _info_text(self):
        """Minimal per-phantom label: index, name, and array shape (no explanatory text)."""
        name = self.names[self.index] if self.index < len(self.names) else ""
        shape = self.history[self.index].shape
        label = f"{self.index + 1}/{len(self.history)}  {name}  shape={shape}"
        if self.is_big[self.index]:
            label += "  [big]"
        return label

    def _make_meshes(self, image_stack: np.ndarray):
        """
        Convert a 3D image stack into PyVista meshes for different regions of interest
        (lump, insert, and pillar).

        Parameters
        ----------
        image_stack : np.ndarray
            A 3D NumPy array (k × h × w) representing the volumetric image stack.

        Returns
        -------
        tuple of pyvista.DataSet
            - lump_mesh : pyvista.UnstructuredGrid
                Mesh extracted by thresholding voxels with value of 255.
            - insert_mesh : pyvista.UnstructuredGrid
                Mesh extracted by thresholding voxels with value of 127.
            - pillar_mesh : pyvista.UnstructuredGrid
                Mesh extracted by thresholding voxels with value of 200.
        """

        k, h, w = image_stack.shape
        grid = pv.ImageData()
        grid.dimensions = image_stack.shape

        original_size = 80

        new_size = w
        new_spacing_y = 0.763889 * (original_size / new_size)
        new_spacing_x = 0.763889 * (original_size / new_size)
        new_spacing_z = 0.9

        if self.names[self.index] in self.big_inserts or self.is_big[self.index]:
            new_spacing_z /= 2

        grid.spacing = (new_spacing_z, new_spacing_y, new_spacing_x)
        grid['values'] = image_stack.flatten(order='F')

        lump_mesh = grid.threshold(value=254)
        insert_mesh = grid.threshold(value=(126, 128))
        pillar_mesh = grid.threshold(value=(199, 201))

        return lump_mesh, insert_mesh, pillar_mesh

    def _reconstruct_tensor_to_np(self, model_images: torch.Tensor) -> np.ndarray:
        """
        Convert a 3D model_images tensor back to a NumPy array with original pixel values.

        Args:
            model_images: torch.Tensor of shape (C, K, H, W) or (K, H, W)
        Returns:
            np.ndarray with shape (K, H, W) with values:
                background = 0
                insert = 127
                pillar = 200
                lump = 255
        """
        if model_images.ndim != 4 and model_images.ndim != 3:
            raise ValueError(f"Expected a 3D tensor (C,K,H,W), got shape {model_images.shape}")

        if model_images.ndim == 4:
            C, K, H, W = model_images.shape

            reconstructed = np.zeros((K, H, W), dtype=np.uint8)  # background = 0

            reconstructed[model_images[1] == 0] = 127
            reconstructed[model_images[2] == 0] = 200
            reconstructed[model_images[3] == 0] = 255
        else:

            K, H, W = model_images.shape
            reconstructed = np.zeros((K, H, W), dtype=np.uint8)  # background = 0

            reconstructed[model_images == 1] = 127
            reconstructed[model_images == 2] = 200
            reconstructed[model_images == 3] = 255

        return reconstructed

    def load_array(self, image_stack, name: str = "", is_big_phantom: bool = False):
        """
          Load a 3D np array or 4D tensor array into history and optionally track its name in debug mode.

          Parameters
          ----------
          image_stack :
              3D array to store can be np.ndarray or a 3d or 4d tensor array.
          name : str, optional.
              Optional label for the array (default "").
        """
        if torch.is_tensor(image_stack):
            image_stack = self._reconstruct_tensor_to_np(image_stack)

        # Ensure 3D
        if len(image_stack.shape) != 3:
            print(f"Error: Array must be 3D. Current shape: {image_stack.shape}")
            return

        self.history.append(image_stack)
        self.is_big.append(is_big_phantom)
        self.names.append(name)

    def load_array_from_path(self, image_stack_path, name=""):
        """
        Load a 3D array from a .npy file and append it to history.

        Parameters
        ----------
        image_stack_path : str or Path.
            Path to the .npy file.
        name : str, optional.
            Optional label for the array (default "").
        """

        try:
            image_stack = np.load(image_stack_path)
        except FileNotFoundError:
            print(f"Error: The file '{image_stack_path}' was not found.")
            return

        self.load_array(image_stack, name=name)

    def load_folder(self, data_path):
        """
        Recursively load all .npy arrays from a folder into history.

        Parameters
        ----------
        data_path : str or Path
            Directory containing .npy files.

        Notes
        -----
        Automatically prints debug info for each loaded file if debug mode is enabled.
        """

        i = 1
        for (dirpath, dirnames, filenames) in os.walk(data_path):
            for f in filenames:
                if f.endswith(".npy"):
                    name = (f.split("_")[0])
                    self.load_array_from_path(os.path.join(dirpath, f), name)

                    if self.debug:
                        print(f"[DEBUG] {i : 3}. Found file: {os.path.join(dirpath, f)}")

                    i += 1

    def smooth_mesh(self, mesh, n_iter=100):
        if mesh is None or mesh.n_points == 0:
            return mesh

        # FIX: Convert UnstructuredGrid to PolyData (surface)
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()

        # Now you can smooth it
        return mesh.smooth(n_iter=n_iter, feature_smoothing=False)

    def _update_scene(self):
        """
        Clear the plotter and render the current array from history.
        """

        if self.index < 0 or self.index >= len(self.history):
            return

        image_stack = self.history[self.index]
        lump_mesh, insert_mesh, pillar_mesh = self._make_meshes(image_stack)

        self.plotter.clear()  # clear old actors

        if self.should_smooth_mesh:
            lump_mesh = self.smooth_mesh(lump_mesh, n_iter=300)
            insert_mesh = self.smooth_mesh(insert_mesh, n_iter=300)
            pillar_mesh = self.smooth_mesh(pillar_mesh, n_iter=300)

        if lump_mesh.n_points > 0:
            self.plotter.add_mesh(
                lump_mesh, color='red', opacity=self.current_lump_opacity, smooth_shading=True
            )

        if insert_mesh.n_points > 0:
            self.gray_actor = self.plotter.add_mesh(
                insert_mesh, color='gray', opacity=self.current_gray_opacity, smooth_shading=True, show_edges=True
            )

        if pillar_mesh.n_points > 0:
            self.plotter.add_mesh(pillar_mesh, color=(240, 240, 240), opacity=self.current_pillar_opacity, smooth_shading=True)

        # Info text
        if self.show_text:
            self.plotter.add_text(f"3D Visualization - {self.name}", font_size=15)
            self.plotter.add_text("Use the mouse to rotate, scroll to zoom, right-click to pan",
                                  font_size=8, position=(0.005, 0.92), viewport=True)
            self.plotter.add_text("Press 'n' for next array, 'b' for previous array",
                                  font_size=8, position=(0.005, 0.89), viewport=True)
            self.plotter.add_text("Press 'r' to reset camera",
                                  font_size=8, position=(0.005, 0.86), viewport=True)
            self.plotter.add_text("Press 'g' to toggle gray mesh opacity",
                                  font_size=8, position=(0.005, 0.83), viewport=True)
            self.plotter.add_text("Press 's' to save screenshot",
                                  font_size=8, position=(0.005, 0.80), viewport=True)
            self.plotter.add_text("Press 'h' to save the visualization as HTML",
                                  font_size=8, position=(0.005, 0.77), viewport=True)
            self.plotter.add_text("Press 'p' to print the camera location",
                                  font_size=8, position=(0.005, 0.74), viewport=True)
            self.plotter.add_text("Press 'q' to quit",
                                  font_size=8, position=(0.005, 0.71), viewport=True)

            # Array progress
            self.plotter.add_text(self._info_text(), font_size=12, position=(0.01, 0.95))

        elif self.show_info:
            # Just the name/shape label, no explanatory instructions
            self.plotter.add_text(self._info_text(), font_size=12, position=(0.01, 0.95))

        self.plotter.render()

        if self.start_flag and self.index not in self.showcased:
            img = self._save_showcase()
            # HTML exports required an interactive GUI
            if not self.off_screen and self.debug:
                self.export_to_html()

            return img

    def reset_history(self):
        self.history = []
        self.index = -1
        self.showcased = set()
        self.names = []

    def toggle_gray(self):
        """
        Cycle the gray mesh opacity through preset values: 0.45 → 0.9 → 1 → 0.3 → 0.45.

        Notes
        -----
        Only affects the gray (insert) mesh. Automatically re-renders the scene.
        """

        if not self.gray_actor:
            return

        current_opacity = self.gray_actor.GetProperty().GetOpacity()
        if current_opacity == self.original_gray_opacity:
            self.gray_actor.GetProperty().SetOpacity(0.9)
        elif current_opacity == 0.9:
            self.gray_actor.GetProperty().SetOpacity(1)
        elif current_opacity == 1:
            self.gray_actor.GetProperty().SetOpacity(0.3)
        else:
            self.gray_actor.GetProperty().SetOpacity(self.original_gray_opacity)

        self.current_gray_opacity = self.gray_actor.GetProperty().GetOpacity()
        self.plotter.render()

    def next_array(self):
        """Go forward in history."""
        if self.index not in self.showcased and self.index == 0:
            self._save_showcase()
            self.export_to_html()

        if self.index < len(self.history) - 1:
            self.index += 1
            self._update_scene()

    def prev_array(self):
        """Go backward in history."""
        if self.index > 0:
            self.index -= 1
            self._update_scene()

    def _save_showcase(self):
        """Save a "showcase" screenshot of the current array using a fixed camera position."""
        self.showcased.add(self.index)

        # temporarily switch camera to showcase position
        old_cpos = self.plotter.camera_position
        self.plotter.camera_position = self.showcase_cpos
        self.plotter.render()  # force apply the camera

        # save screenshot at showcase view
        if self.index % 2 == 0:
            idx = int(self.index / 2)
            filename = self.showcase_path / f"{self.name}_showcase_gt_{idx}.png"
        else:
            idx = int(self.index / 2)
            filename = self.showcase_path / f"{self.name}_showcase_pred_{idx}.png"

        img = self.plotter.screenshot(str(filename), return_img=True)

        if self.debug:
            print(f"[DEBUG] Saved showcase screenshot: {filename}")

        # restore old camera
        self.plotter.camera_position = old_cpos
        self.plotter.render()

        return img

    def reset_camera(self):
        """Reset the camera to the default position."""
        self.plotter.camera_position = self.default_cpos

    def save_screenshot(self):
        """Save a PNG screenshot of the current scene."""
        self.screenshot_counter += 1
        filename = self.output_dir / f"{self.name}_screenshot_{self.screenshot_counter:03d}.png"
        self.plotter.screenshot(str(filename))

        if self.debug:
            print(f"[DEBUG] Saved screenshot: {filename}")

    def export_to_html(self, filename=None):
        """
        Export the current 3D scene to an interactive HTML file.

        Parameters
        ----------
        filename : str or Path, optional
            Specific filename for the HTML file. If None, a default name is used.
        """

        if filename is None:
            if self.index % 2 == 0:
                idx = int(self.index / 2)
                filename = self.html_path / f"{self.name}_html_gt_{idx}.html"
            else:
                idx = int(self.index / 2)
                filename = self.html_path / f"{self.name}_html_pred_{idx}.html"
        else:
            filename = Path(filename)
            filename = self.html_path / filename.with_suffix(".html")

        # Export the HTML visualization
        self.plotter.export_html(str(filename))

        # Add the array index to the generated HTML file
        if self.show_text:
            try:
                with open(filename, 'r') as file:
                    content = file.read()

                new_content = content.replace(
                    "<body>",
                    f"<body>\n    <p style='position:absolute; top: 10px; left: 10px; "
                    f"z-index: 10; font-family:sans-serif; font-size: 20px;'>"
                    f"Array {self.index + 1}</p>"
                )

                with open(filename, 'w') as file:
                    file.write(new_content)

                if self.debug:
                    print(f"[DEBUG] Saved HTML visualization with text to: {filename}")
            except FileNotFoundError:
                print(f"Error: The HTML file '{filename}' was not found after export.")
        else:
            if self.debug:
                print(f"[DEBUG] Saved HTML visualization without overlay to: {filename}")

    def print_camera(self):
        """Print the current camera position to the console."""
        print("\nCamera position:", self.plotter.camera_position)

    def quit_program(self):
        """Close the visualization window"""
        self.plotter.close()

        if self.debug:
            print("[DEBUG] Finished visualization")

    def show(self, cpos: Optional[CameraPosition] = None):
        """
        Start the interactive GUI visualization loop.

        Parameters
        ----------
        cpos : tuple, optional
            Camera position to use when starting. Defaults to `self.default_cpos`.

        Raises
        ------
        RuntimeError
            If no arrays have been loaded into history.
        """
        if not self.off_screen:
            if len(self.history) == 0:
                raise RuntimeError("No array to render")

            # Start at the first frame
            self.index = 0
            self._update_scene()
            self.start_flag = True

            self.plotter.show(auto_close=False, cpos=cpos or self.default_cpos)
        else:
            imgs = self.render_all()
            return imgs

    def render_all(self):
        """Loop through history and save screenshots + HTML for each array."""
        if len(self.history) == 0:
            return

        imgs = []
        self.start_flag = True
        for i in range(len(self.history)):
            self.index = i
            img = self._update_scene()
            imgs.append(img)

        return imgs
