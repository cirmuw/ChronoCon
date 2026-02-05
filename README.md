## Chronological Contrastive Learning: Few-Shot Progression Assessment in Irreversible Diseases
This repository contains the relencant code for the MIDL 2026 submission.

The ChronoCon code was written by Clemens Watzenböck. The original version of the preprocessing code (joint-detetection, automatic ROI size adaptation, patch extraction, ...) was written by Thomas Deimel and Paul Weiser.
It is also contained in this repository. More details in the dataset and prepocessing can be found in [DOI:10.64898/2025.12.26.25343056; see reference below]. Note that the preprocessing code used for the RA-dataset (point-annotations + landmark detection) is in a seperate repository [landmarks_utils](https://github.com/cirmuw/autopix_landmarks_utils). 


### Some key components
- ChronoCon loss function: ["RnCLossMono"](./ra_utils/loss/loss_RnCMono.py)
- Main script for training: [train.py](./ra_utils/training/scores_SHS/07_train.py). 
- Sample input script [config.yml](./runs/config_scoring/development_inputs/training_config_cleaned.yml)

These two should be called as 
```bash 
python train.py  --config config.yml
```


## 🚀 Installation (via conda)

Follow these steps to set up your environment:

```bash
# Step 1: Install CUDA (if using GPU)
# Follow official instructions from: https://developer.nvidia.com/cuda-downloads

# Step 2: Create and activate a conda environment. E.g. 
conda create -n py3-10-ra_utils python=3.10
conda activate py3-10-ra_utils

# Step 3: Install this package (ra_utils)
pip install -e .
```



### TODO
- [ ] Clean up and remove old landmark detection code (now in [landmarks_utils](https://github.com/cirmuw/autopix_landmarks_utils) anyhow. 
- [ ] Remove old no training scripts, and other old code. 
- [ ] Add more extensive documentation on preprocessing and related topics at a later stage
- [ ] Add reference to my MIDL paper(submission)  🤞 


### References

```bash
# [My MIDL 2026 submission placeholder 🤞]
@unpublished{watzenboeck2026chronocon,
  title     = {Chronological Contrastive Learning: Few-Shot Progression Assessment in Irreversible Diseases},
  author    = {Watzenb{\"o}ck, Clemens and Aletaha, Daniel and Deman, Micha{\"e}l and Deimel, Thomas and Eder, Jana and Janickova, Ivana and Janiczek, Robert and Mandl, Peter and Seeb{\"o}ck, Philipp and Supp, Gabriela and Weiser, Paul and Langs, Georg},
  note      = {Submitted to MIDL 2026},
  year      = {2026},
}


# [Related work: Dataset reference]
@article{deimel2025autoscoRA,
	author = {Deimel, Thomas and Weiser, Paul J. and Urschler, Martin and Payer, Christian and Mandl, Peter and Langs, Georg and Aletaha, Daniel},
	title = {autoscoRA: Deep Learning to Automate Sharp/van der Heijde Scoring of Radiographic Damage in Rheumatoid Arthritis},
	elocation-id = {2025.12.26.25343056},
	year = {2025},
	doi = {10.64898/2025.12.26.25343056},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/12/29/2025.12.26.25343056},
	eprint = {https://www.medrxiv.org/content/early/2025/12/29/2025.12.26.25343056.full.pdf},
	journal = {medRxiv}
}
```



### Other notes: 
#### `.env`: 
 - I started experimenting with dinov3 as backbone. It was usefull to have the dinov3 cloned locally and also to have the official pretrained weights stored locally. See `.env` file.


```bash 

DINOV3_WEIGHTS_LOCATION="/home/cwatzenboeck/data/dinov3/weights"
DINOV3_CODE_DIR="/home/cwatzenboeck/code/RA/public/dinov3/"

```


