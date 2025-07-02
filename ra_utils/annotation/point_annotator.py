from typing import List

from dask_image.imread import imread
from magicgui.widgets import ComboBox, Container
import napari
import pandas as pd
import pydicom
import numpy as np
from qtpy.QtCore import QTimer   

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
            "size": 8,                       
            "color": "blue", 
            "anchor": "upper_left",           
            "translation": np.array([10, 10]), 
        },
    )
    points_layer.border_color_mode = "cycle"


    # --- Label-selection widget ---
    label_widget = create_label_menu(points_layer, labels)
    viewer.window.add_dock_widget(label_widget)


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

