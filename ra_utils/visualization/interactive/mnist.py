import fiftyone as fo
import fiftyone.zoo as foz



import cv2
import numpy as np

import fiftyone.brain as fob

import cv2
import numpy as np

import fiftyone.brain as fob

def main():

    dataset = foz.load_zoo_dataset("mnist")
    test_split = dataset.match_tags("test")

    # Construct a ``num_samples x num_pixels`` array of images
    embeddings = np.array([
        cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel()
        for f in test_split.values("filepath")
    ])

    # Compute 2D representation
    results = fob.compute_visualization(
        test_split,
        embeddings=embeddings,
        num_dims=2,
        method="umap",
        brain_key="mnist_test",
        verbose=True,
        seed=51,
    )



    dataset.load_brain_results("mnist_test")
    session = fo.launch_app(test_split)


if __name__ == "__main__":
    main()
