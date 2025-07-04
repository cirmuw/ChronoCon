from typing import List

from dask_image.imread import imread
from magicgui.widgets import ComboBox, Container
import napari
import pandas as pd
import pydicom
import numpy as np
from qtpy.QtCore import QTimer   
from pathlib import Path
from typing import List, Sequence, Union

COLOR_CYCLE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]


def create_label_menu(points_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock

    Parameters
    ----------
    points_layer : napari.layers.Points
        a napari points layer
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).

    Returns
    -------
    label_menu : Container
        the magicgui Container with our dropdown menu widget
    """
    # Create the label selection menu
    label_menu = ComboBox(label='feature_label', choices=labels)
    label_widget = Container(widgets=[label_menu])


    def update_label_menu(event):
        """Update the label menu when the point selection changes"""
        new_label = str(points_layer.feature_defaults['label'][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    points_layer.events.feature_defaults.connect(update_label_menu)

    def label_changed(selected_label):
        """Update the Points layer when the label menu selection changes"""
        feature_defaults = points_layer.feature_defaults
        feature_defaults['label'] = selected_label
        points_layer.feature_defaults = feature_defaults
        points_layer.refresh_colors()

    label_menu.changed.connect(label_changed)

    return label_widget




def point_annotator(
        dicom_path: str,
        labels: List[str],
        save_dst_dir: str = None
):
    """Annotate landmarks in a single DICOM X-ray.

    Parameters
    ----------
    dicom_path : str
        Path to a single `.dcm` file.
    labels : List[str]
        Ordered list of landmark names.
    """
    # --- Load DICOM ---
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))



    # === INIT VIEWER ===
    viewer = napari.view_image(img, name="X-ray", colormap="gray")    

    # --- Empty points layer (2-D) ---
    points_layer = viewer.add_points(
        ndim=2,
        features=pd.DataFrame(
            {"label": pd.Categorical([], categories=labels)}
        ),
        border_color="label",
        border_color_cycle=COLOR_CYCLE,
        symbol="o",
        face_color="transparent",
        border_width=0.5,
        size=12,
        text={
            "string": "{label}",               
            "size": 14,                       
            "color": "cyan", 
            "anchor": "upper_left",           
            "translation": np.array([10, 10]), 
        },
    )
    points_layer.border_color_mode = "cycle"


    # --- Label-selection widget ---
    label_widget = create_label_menu(points_layer, labels)
    viewer.window.add_dock_widget(label_widget)

    if save_dst_dir is None:
        dicom_path = Path(dicom_path)                  # make sure it's a Path object
        csv_path  = str(dicom_path) + '__landmarks.csv'    # e.g.  hand_xray.dcm → hand_xray.csv    
    else:
        file_name = Path(dicom_path).name
        csv_path = str(Path(save_dst_dir) / (file_name + '__landmarks.csv'))
        
    def _save_csv(layer):
        if len(layer.data) == 0:
            viewer.status = 'No points to save …'
            return

        df = pd.DataFrame(layer.data, columns=['y', 'x'])   # row, col → y, x
        df['label'] = layer.features['label'].to_numpy()
        df["file_name"] = Path(dicom_path).name
        df["dicom_path"] = str(dicom_path)

        try:
            df.to_csv(csv_path, index=False)
            viewer.status = f'Saved {len(df)} landmarks → {csv_path}'
        except Exception as e:
            viewer.status = f'❌  Could not save CSV: {e}'



    # after points_layer is created
    @points_layer.bind_key('s', overwrite=True)     # captures the key before the layer does
    def save_points(layer):                         # “layer” == points_layer
        _save_csv(layer)

    def _cycle(n=1):
        """Advance the default label by *n*, wrapping around."""
        f = points_layer.feature_defaults
        idx = labels.index(f['label'][0])
        f['label'] = labels[(idx + n) % len(labels)]
        points_layer.feature_defaults = f
        points_layer.refresh_colors()


    @viewer.bind_key('.')          # manual next / previous remain unchanged
    def next_label(event=None): _cycle(+1)

    @viewer.bind_key(',')
    def prev_label(event=None): _cycle(-1)

    # ----------------------------------------------------------------------
    def advance_on_click(layer, event):
        """Cycle the default label **after** a click has created a point."""
        if layer.mode != 'add':
            return

        layer.selected_data = set()     # keep the new point un-selected

        yield                           # ------------ mouse pressed ----------

        # if the user drags, we keep yielding until the button comes up
        while event.type == 'mouse_move':
            yield

        # now we're in the *mouse-release* event → point already exists
        # but we still delay the cycle with a 0-ms QTimer so napari
        # finishes its own housekeeping first.
        QTimer.singleShot(0, lambda: _cycle(+1))

    # attach the callback
    points_layer.mode = 'add'
    points_layer.mouse_drag_callbacks.append(advance_on_click)

    napari.run()





# def point_annotator_multi(
#     dicom_paths: Union[str, Path, Sequence[Union[str, Path]]],
#     labels: List[str],
#     save_dst_dir: Union[str, Path, None] = None,
#     df_double_scoring: Union[pd.DataFrame, None]= None
# ):
#     """Annotate landmarks in one **or more** DICOM radiographs.

#     Key bindings (viewer‑wide):
#         • **Right Arrow**  – next image
#         • **Left Arrow**   – previous image
#         • **n / p**        – next / previous image (alternative)
#         • **s**            – save landmarks for current image
#         • **r**            – reload landmarks for current image
#         • **. / ,**        – cycle labels forward / backward

#     Landmarks auto‑save before every navigation so you never lose work.
#     """

#     # ------------------------------------------------------------------
#     # normalise input ---------------------------------------------------
#     if isinstance(dicom_paths, (str, Path)):
#         dicom_paths = [dicom_paths]
#     dicom_paths = [Path(p) for p in dicom_paths]
#     if not dicom_paths:
#         raise ValueError("No DICOM paths provided.")
#     n_images = len(dicom_paths)

#     # ------------------------------------------------------------------
#     # helpers -----------------------------------------------------------
#     def _read_dicom(path: Path, use_quantile_norm: bool = True) -> np.ndarray:
#         ds = pydicom.dcmread(str(path))
#         img = ds.pixel_array.astype(np.float32)

#         if use_quantile_norm:
#             lower = np.percentile(img, 1)
#             upper = np.percentile(img, 99)
#         else:
#             lower = img.min()
#             upper = img.max()

#         img = (img - lower) / (upper - lower + 1e-6)
#         img = np.clip(img, 0, 1)  # ensure values stay in [0, 1] range

#         return img

#     def _csv_path_for(path: Path) -> Path:
#         if save_dst_dir is None:
#             return path.with_suffix(path.suffix + "__landmarks.csv")
#         return Path(save_dst_dir) / f"{path.name}__landmarks.csv"

#     # ------------------------------------------------------------------
#     # viewer + layers ---------------------------------------------------
#     current_idx = 0  # will mutate inside _goto

#     viewer = napari.view_image(
#         _read_dicom(dicom_paths[current_idx]),
#         name=f"X‑ray (1/{n_images})",
#         colormap="gray",
#     )
#     image_layer = viewer.layers[0]

#     points_layer = viewer.add_points(
#         ndim=2,
#         features=pd.DataFrame({"label": pd.Categorical([], categories=labels)}),
#         border_color="label",
#         border_color_cycle=COLOR_CYCLE,
#         symbol="o",
#         face_color="transparent",
#         border_width=0.5,
#         size=12,
#         text={
#             "string": "{label}",
#             "size": 14,
#             "color": "cyan",
#             "anchor": "upper_left",
#             "translation": np.array([10, 10]),
#         },
#     )
#     points_layer.border_color_mode = "cycle"
#     points_layer.mode = "add"

#     # ------------------------------------------------------------------
#     # persistence -------------------------------------------------------
#     def _save_csv():
#         if len(points_layer.data) == 0:
#             return
#         df = pd.DataFrame(points_layer.data, columns=["y", "x"])
#         df["label"] = points_layer.features["label"].to_numpy()
#         df["file_name"] = dicom_paths[current_idx].name
#         df["dicom_path"] = str(dicom_paths[current_idx])
#         csv_path = _csv_path_for(dicom_paths[current_idx])
#         csv_path.parent.mkdir(parents=True, exist_ok=True)
#         df.to_csv(csv_path, index=False)
#         viewer.status = f"Saved {len(df)} landmarks → {csv_path}"

#     def _load_csv():
#         #viewer.status = f"Loading landmarks ({current_idx + 1}/{n_images}) …"
#         csv_path = _csv_path_for(dicom_paths[current_idx])
#         if csv_path.exists():
#             df = pd.read_csv(csv_path)
#             points_layer.data = df[["y", "x"]].to_numpy()
#             points_layer.features = pd.DataFrame(
#                 {"label": pd.Categorical(df["label"], categories=labels)}
#             )
#             viewer.status = f"Loaded {len(df)} landmarks from {csv_path}"
#         else:
#             points_layer.data = np.empty((0, 2))
#             points_layer.features = pd.DataFrame(
#                 {"label": pd.Categorical([], categories=labels)}
#             )
#             viewer.status = f"No landmarks for {dicom_paths[current_idx].name}"
#         points_layer.refresh()

#     print("Loading landmarks for the first image …")
#     _load_csv()

#     # ------------------------------------------------------------------
#     # navigation --------------------------------------------------------
#     def _goto(idx: int):
#         nonlocal current_idx
#         _save_csv()
#         current_idx = idx % n_images
#         viewer.status = f"Loading image for ({current_idx + 1}/{n_images}) …"
#         image_layer.data = _read_dicom(dicom_paths[current_idx])
#         image_layer.name = f"X‑ray ({current_idx + 1}/{n_images})"
#         _load_csv()

#     # viewer‑level bindings (most reliable; apply regardless of focus)
#     @viewer.bind_key("Right", overwrite=True)
#     def _next(viewer):  # viewer is passed automatically
#         _goto(current_idx + 1)

#     @viewer.bind_key("Left", overwrite=True)
#     def _prev(viewer):
#         _goto(current_idx - 1)

#     # alternative letter shortcuts (also viewer‑wide)
#     @viewer.bind_key("n", overwrite=True)
#     def _next_n(viewer):
#         print("Next image")
#         viewer.status = "Next image"
#         _goto(current_idx + 1)

#     @viewer.bind_key("p", overwrite=True)
#     def _prev_p(viewer):
#         _goto(current_idx - 1)

#     # save / reload (viewer‑wide so they work in any context)
#     #@viewer.bind_key("s", overwrite=True)
#     @points_layer.bind_key('s', overwrite=True)     # captures the key before the layer does
#     def _save(viewer):
#         _save_csv()

#     # @viewer.bind_key("r", overwrite=True)
#     @points_layer.bind_key('s', overwrite=True)
#     def _reload(viewer):
#         _load_csv()

#     # ------------------------------------------------------------------
#     # label cycling -----------------------------------------------------
#     def _cycle(n=1):
#         f = points_layer.feature_defaults
#         idx = labels.index(f["label"][0]) if len(f["label"]) else 0
#         f["label"] = labels[(idx + n) % len(labels)]
#         points_layer.feature_defaults = f
#         points_layer.refresh_colors()

#     @viewer.bind_key(".")
#     def _next_label(viewer):
#         _cycle(+1)

#     @viewer.bind_key(",")
#     def _prev_label(viewer):
#         _cycle(-1)

#     # ------------------------------------------------------------------
#     # auto‑advance label on click --------------------------------------
#     def advance_on_click(layer, event):
#         if layer.mode != "add":
#             return
#         layer.selected_data = set()
#         yield  # press
#         while event.type == "mouse_move":
#             yield
#         QTimer.singleShot(0, lambda: _cycle(+1))

#     points_layer.mouse_drag_callbacks.append(advance_on_click)

#     napari.run()



def point_annotator_multi(
    dicom_paths: Union[str, Path, Sequence[Union[str, Path]]],
    labels: List[str],
    save_dst_dir: Union[str, Path, None] = None,
    df_double_scoring: Union[pd.DataFrame, None] = None,
):
    """Annotate landmarks in one **or more** DICOM radiographs.

    Parameters
    ----------
    dicom_paths
        One path or a sequence of paths pointing to DICOM files.
    labels
        List of landmark labels (category order defines colour‑cycle order).
    save_dst_dir
        Optional directory where landmark CSV files are written. If *None* the
        CSV is stored next to each DICOM file.
    df_double_scoring
        *Optional* DataFrame holding **ground‑truth** landmark coordinates to
        visualise for reference (a.k.a. *double scoring*). Must contain:

        * ``file_name`` (or ``dicom_path``)
        * ``y`` and ``x`` columns (row/column order as usual for napari)
        * ``label`` column matching *labels*

        If *df_double_scoring* is *None* no ground‑truth layer is displayed.

    Key bindings (viewer‑wide)
    --------------------------
    • **Right Arrow**  – next image
    • **Left Arrow**   – previous image
    • **n / p**        – next / previous image (alternative)
    • **s**            – save landmarks for current image
    • **r**            – reload landmarks for current image
    • **. / ,**        – cycle labels forward / backward
    • **g**            – toggle ground‑truth visibility (if available)

    Landmarks auto‑save before every navigation so you never lose work.
    """

    labels = [ f"{i} {name}" for i, name in enumerate(labels) ]

    # ────────────────────────────────────────────────────────────────────
    # normalise input                                                   
    # ────────────────────────────────────────────────────────────────────
    if isinstance(dicom_paths, (str, Path)):
        dicom_paths = [dicom_paths]
    dicom_paths = [Path(p) for p in dicom_paths]
    if not dicom_paths:
        raise ValueError("No DICOM paths provided.")
    n_images = len(dicom_paths)

    # ────────────────────────────────────────────────────────────────────
    # helpers                                                           
    # ────────────────────────────────────────────────────────────────────
    def _read_dicom(path: Path, use_quantile_norm: bool = True) -> np.ndarray:
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)

        if use_quantile_norm:
            lower = np.percentile(img, 1)
            upper = np.percentile(img, 99)
        else:
            lower = img.min()
            upper = img.max()

        img = (img - lower) / (upper - lower + 1.0e-6)
        return np.clip(img, 0, 1)

    def _csv_path_for(path: Path) -> Path:
        if save_dst_dir is None:
            return path.with_suffix(path.suffix + "__landmarks.csv")
        return Path(save_dst_dir) / f"{path.name}__landmarks.csv"

    # ────────────────────────────────────────────────────────────────────
    # viewer + layers                                                   
    # ────────────────────────────────────────────────────────────────────
    current_idx = 0  # will mutate inside _goto

    viewer = napari.view_image(
        _read_dicom(dicom_paths[current_idx]),
        name=f"X‑ray (1/{n_images})",
        colormap="gray",
    )
    image_layer = viewer.layers[0]

    # editable points (what the annotator is drawing)
    points_layer = viewer.add_points(
        ndim=2,
        features=pd.DataFrame({"label": pd.Categorical([], categories=labels)}),
        border_color="label",
        border_color_cycle=COLOR_CYCLE,
        symbol="o",
        face_color="transparent",
        border_width=0.5,
        size=12,
        text={
            "string": "{label}",
            "size": 14,
            "color": "cyan",
            "anchor": "upper_left",
            "translation": np.array([10, 10]),
        },
        name="Annotations",
    )
    points_layer.border_color_mode = "cycle"
    points_layer.mode = "add"


    # --- Label-selection widget ---
    label_widget = create_label_menu(points_layer, labels)
    viewer.window.add_dock_widget(label_widget)

    # non‑editable ground‑truth reference layer (optional)
    if df_double_scoring is not None:
        ground_truth_layer = viewer.add_points(
            ndim=2,
            symbol="x",
            size=20,
            border_color="red",  # sunflower yellow
            face_color="transparent",
            opacity=0.8,
            name="Ground‑truth",
            visible=True,
        )
        ground_truth_layer.editable = False
        ground_truth_layer.interactive = False
    else:
        ground_truth_layer = None  # keeps typing simple below

    # ────────────────────────────────────────────────────────────────────
    # persistence                                                       
    # ────────────────────────────────────────────────────────────────────
    def _save_csv():
        if len(points_layer.data) == 0:
            return
        df = pd.DataFrame(points_layer.data, columns=["y", "x"])
        df["label"] = points_layer.features["label"].to_numpy()
        df["file_name"] = dicom_paths[current_idx].name
        df["dicom_path"] = str(dicom_paths[current_idx])
        csv_path = _csv_path_for(dicom_paths[current_idx])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        viewer.status = f"Saved {len(df)} landmarks → {csv_path}"

    def _load_csv():
        """Load user annotations *and* optional ground‑truth for current image."""
        csv_path = _csv_path_for(dicom_paths[current_idx])
        # ── user annotations ────────────────────────────────────────────
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            points_layer.data = df[["y", "x"]].to_numpy()
            points_layer.features = pd.DataFrame(
                {"label": pd.Categorical(df["label"], categories=labels)}
            )
            viewer.status = f"Loaded {len(df)} landmarks from {csv_path}"
        else:
            points_layer.data = np.empty((0, 2))
            points_layer.features = pd.DataFrame(
                {"label": pd.Categorical([], categories=labels)}
            )
            viewer.status = f"No landmarks for {dicom_paths[current_idx].name}"
        points_layer.refresh()

        # ── ground‑truth reference ──────────────────────────────────────
        if ground_truth_layer is None:
            return

        df_gt = None
        if "dicom_path" in df_double_scoring.columns:
            df_gt = df_double_scoring[
                df_double_scoring["dicom_path"] == str(dicom_paths[current_idx])
            ]
        if (df_gt is None or df_gt.empty) and "file_name" in df_double_scoring.columns:
            df_gt = df_double_scoring[
                df_double_scoring["file_name"] == dicom_paths[current_idx].name
            ]
        if df_gt is not None and not df_gt.empty:
            ground_truth_layer.data = df_gt[["y", "x"]].to_numpy()
            ground_truth_layer.features = pd.DataFrame(
                {"label": pd.Categorical(df_gt["label"], categories=labels)}
            )
        else:
            ground_truth_layer.data = np.empty((0, 2))
            ground_truth_layer.features = pd.DataFrame(
                {"label": pd.Categorical([], categories=labels)}
            )
        ground_truth_layer.refresh()

    # initial load
    print("Loading landmarks for the first image …")
    _load_csv()

    # ────────────────────────────────────────────────────────────────────
    # navigation                                                        
    # ────────────────────────────────────────────────────────────────────
    def _goto(idx: int):
        nonlocal current_idx
        _save_csv()
        current_idx = idx % n_images
        viewer.status = f"Loading image ({current_idx + 1}/{n_images}) …"
        image_layer.data = _read_dicom(dicom_paths[current_idx])
        image_layer.name = f"X‑ray ({current_idx + 1}/{n_images})"
        _load_csv()

    @viewer.bind_key("Right", overwrite=True)
    def _next(viewer):
        viewer.status = "Loading next image"
        _goto(current_idx + 1)

    @viewer.bind_key("Left", overwrite=True)
    def _prev(viewer):
        viewer.status = "Loading previous image"
        _goto(current_idx - 1)

    @viewer.bind_key("n", overwrite=True)
    def _next_n(viewer):
        viewer.status = "Loading next image"
        _goto(current_idx + 1)

    @viewer.bind_key("p", overwrite=True)
    def _prev_p(viewer):
        _goto(current_idx - 1)

    # save / reload -----------------------------------------------------
    @points_layer.bind_key("s", overwrite=True)
    def _save(viewer):
        _save_csv()

    @points_layer.bind_key("r", overwrite=True)
    def _reload(viewer):
        _load_csv()

    # ────────────────────────────────────────────────────────────────────
    # label cycling                                                     
    # ────────────────────────────────────────────────────────────────────
    def _cycle(n: int = 1):
        f = points_layer.feature_defaults
        idx = labels.index(f["label"][0]) if len(f["label"]) else 0
        f["label"] = labels[(idx + n) % len(labels)]
        points_layer.feature_defaults = f
        points_layer.refresh_colors()

    @viewer.bind_key(".")
    def _next_label(viewer):
        _cycle(+1)

    @viewer.bind_key(",")
    def _prev_label(viewer):
        _cycle(-1)

    # ────────────────────────────────────────────────────────────────────
    # ground‑truth visibility toggle                                    
    # ────────────────────────────────────────────────────────────────────
    if ground_truth_layer is not None:

        @viewer.bind_key("g")
        def _toggle_gt(viewer):
            ground_truth_layer.visible = not ground_truth_layer.visible
            viewer.status = (
                "Ground‑truth shown" if ground_truth_layer.visible else "Ground‑truth hidden"
            )

    # ────────────────────────────────────────────────────────────────────
    # auto‑advance label on click                                       
    # ────────────────────────────────────────────────────────────────────
    def advance_on_click(layer, event):
        if layer.mode != "add":
            return
        layer.selected_data = set()
        yield  # press
        while event.type == "mouse_move":
            yield
        QTimer.singleShot(0, lambda: _cycle(+1))

    points_layer.mouse_drag_callbacks.append(advance_on_click)

    napari.run()
