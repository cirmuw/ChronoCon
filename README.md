### Legacy info: 

We used to have a different structure, with a different design philosophy, which lead us to the structure: 
```bash 
.
└── libs
    ├── external_packages
    │   └── ext_package_A
    └── ra_utils
        ├── ra_utils
        │   ├── ...
        ...
```

This structure has several downsides. (E.g. it is not simply installable via `pip install git+https//URL_OF_REPO.git` which makes building of containers on the cluster  unnecessarily complicated). It is also not the most common use case. The standard case is:

*One* git repository *per* package. 

Which is now reflected in the new structure. 

Nonetheless, the old structure is still available in the 
[`multi_package_project`](https://gitlab.cir.meduniwien.ac.at:8888/cwatzenboeck/project_structure_template01/-/tree/multi_package_project?ref_type=heads) branch.



# project_structure_template01

This is a template how a project structure might look like. It was created with having a python project in mind. For a c++ project some adaptations are surely needed. 
Some of the highlights included: 

- The project contained an example python package which is easily installable with `pip install -e` 
- CI pipeline is already set up to automatically run unit- and doc-tests with `pytest`
- CI pipeline is already set up to automatically build a documentation as an `html`

Feel free to change it to your needs as necessary. 
When you make changes to this structure, which might make sense to for others as well, please create a merge request. 

*Note:* If you want to use Unit-tests and so in in the gitlab pipeline, you will need to add a gitlab runner to your repository. Otherwise these will not work in the CI-CD-pipeline. Just google it if you need it. Or, if you want to remove this feature you can also delete the [`.gitlab-ci.yml`](.gitlab-ci.yml`) file in your fork of the repo. 


**Confluence**: There is corresponding confluence page for this template. [`https://confluence.meduniwien.ac.at/display/CIRLAB/Project+Structure`](https://confluence.meduniwien.ac.at/display/CIRLAB/Project+Structure).
If you make major chages here consider updating the description on the confluence page.

## Install instructions: 

The python packages come with an appropriate `pyproject.toml` file, which specifies the dependencies. 
This allows you to install them simply with

```
pip install -e .
```

Or, if you don't need a version to edit, but just want to use the code, you can use: 
```bash
pip install git+ssh://git@gitlab.cir.meduniwien.ac.at:11122/cwatzenboeck/project_structure_template01.git
```

For instructions on how to build it via `pip install git+https ...` in a singularity container (on the MSC cluster!) see: 

[`singularity/workflow_a_la_konstantin`](singularity/workflow_a_la_konstantin). You might want to take a look at [`sample_file.sif`](singularity/workflow_a_la_konstantin/sample_file.sif).
It boils down to creating an access token and modifying `sample_file.sif` accordingly. This way you can build a container on the cluster directly and all the dependecies are taken care of by putting them into `pyproject.toml`.

### General structure

```bash
# Run this to take a look at the folder structure 
tree -I '*.pyc|__pycache__|ra_utils.*.rst|*.egg-info|.git|build|html|.env|.gitkeep|_static|_templates|ra_utils.rst|.pytest_cache|tmp' -a 
```

```bash
.
├── ra_utils
│   ├── __init__.py
│   ├── data
│   │   ├── dataloader.py
│   │   └── download_dataset001_liver.py
│   ├── features
│   │   └── transforms.py
│   ├── networks
│   │   ├── architecture.py
│   │   ├── layers.py
│   │   └── loss_function.py
│   ├── utils
│   │   ├── example_module_illustrating_docstrings.py
│   │   └── example_module_illustrating_pytest__test.py
│   └── visualization
│       └── semantic_segmentation.py
├── docs
│   ├── make.bat
│   ├── Makefile
│   └── source
│       ├── conf.py
│       └── index.rst
├── documentation
├── singularity
│   ├── example_project.def
│   ├── example_project_MSC_Philip_Meixner.def
│   ├── example_project_MSC_Philip_Meixner.job
│   └── workflow_a_la_konstantin
│       ├── build_container_on_msc_cluster.job
│       ├── README.md
│       ├── run_arbitrary_command_with_container.job
│       └── sample_file.def
├── tutorials
│   └── preprocessing_pipline.ipynb
├── pyproject.toml
├── .env-example
├── .gitignore
├── .gitlab-ci.yml
├── logs
├── notebooks
├── README.md
├── references
├── requirements.txt
├── slurm
├── tests
└── LICENSE
```


[Here](https://docs.google.com/document/d/1_USwRnq4MR6dlpCHX_2xd5flOwr3TpvJ9h362sOBOFg/edit) you can find more infomation regarding each of the folders and files.



_________________

## TODOS
We are getting there, but some parts can definitely be improved.

### Ongoing
- [ ] CI pipeline 
  - [x] Basic testing with `pytest` in CI pipeline
  - [x] Automatically build documentation 
    - [ ] The structure of the package is not well reflected in the documentation 
  - [ ] Add an official gitlab-runner: 
       Currently the CI pipeline runs on my personal `cirpc`. There must be a better solution for this. 
       Maybe after we move to a different gitlab this will take care of itself. :)
  - [ ] Speed up CI pipeline
        Building the environment is currently slow if your project depends large packages like PyTorch. 
        There are several options to fix this. E.g. caching or building and deploying docker container. 
        My first attempt (caching the virtual environment) did not speed up pipeline
  - [ ] Deploy documentation (where to?)
        The documentation is automatically build in the CI pipeline :-), however, one can only download it from the artifacts. 
        Ideally the built html would also be deployed to some webpage (e.g. gitlab page) so that one can simply look up the 
        API reference, ... in the browser.  
- [ ] *Logging*: We set up a folder `logs` where some logging mechanism (``mlflow`?) might put log files. This should be aligned
      the logging team. 
  - [ ] Log environment (where?, how?, in pipeline?). Automatically save the environment for each commit.
- [x] *singularity*: We added a simple example how a singularity image might be build. (See [`singularity/workflow_a_la_konstantin/sample_file.sif`](singularity/workflow_a_la_konstantin/sample_file.sif) and also the readme in the same folder).
      However, this can and should be improved. 
- [ ] Clean up: 
      I added some code to test building the documentation, ... . Some parts should be cleaned up, others removed. 
- [ ] Fix duplicate dependencies: 
      Currently the projects dependencies are in [requirements.txt](requirements.txt) and also in the packages [pyproject.toml](pyproject.toml) file. 
      This is not ideal. When one adds a dependency, one has to do it in two places. Maybe one should improve this. 
      On the other hand, a project might contain several packages with different requirements. 
- [x] Freeze environment where documentation build works as a checkpoint.  




### Completed Column ✓
- [x] Set up the basic structure
- [x] Add a simple [tutorial example](tutorials/preprocessing_pipline.ipynb) for the package.
- [x] Add examples how to write docstrings: [ra_utils/utils/example_module_illustrating_docstrings.py](ra_utils/utils/example_module_illustrating_docstrings.py)
- [x] Add examples how to write test for your module: [ra_utils/utils/example_module_illustrating_pytest__test.py](ra_utils/utils/example_module_illustrating_pytest__test.py)


_________________


## Example `.env`
Environment variables should be put in the `.env` file, which is included in the `.gitignore.
For an example the variable names used in this project see: [`.env-example`](.env-example). You may copy this file as `.env` and adapt it to your needs.  
Remember: Do NOT commit your environment variables to git!

```bash
SUPER_SECRET_API_KEY=....

```


They can be used in your python scripts in the following way: 
```python

# OPTION 1:  
# Load environment variables from .env file if it exists 
from dotenv import load_dotenv
load_dotenv()

# OPTION 2:
from dotenv import dotenv_values

config = dotenv_values(".env"),  # load environment variables as dictionary

# Note that this allows the management of complex environments. E.g. 
import os
from dotenv import dotenv_values

config = {
    **dotenv_values(".env.shared"),  # load shared development variables
    **dotenv_values(".env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}
```

_________________


## How to access documentation
### Automatically built documentation
1.  Go to the latest pipeline test (click on the little check next to the commit id).
2.  Click on build_docs.
3.  On the right column, press "Download".
4.  Unzip the downloaded file. 
5.  Navigate to the html folder (usually the path looks like this: /artifacts/libs/package_name/docs/build/html)
6.  Open the index.html file

### Build documentation locally in your device
1. Clone the repository.
2. Make sure the environment you are working on to build the documentation has the necessary packages installed. You can find the list in the [docs_requirement.txt](https://drive.google.com/file/d/1PJCuWH5fsyVhBVmIesuaV0p3lSmoshLp/view?usp=sharing) provided. 
3. Navigate to the project_structure_template01\libs\ra_utils\docs folder.
4. Run the following command: sphinx-build -M html ./source ./
5. You will find the documentation built in the project_structure_template01\libs\ra_utils\docs\folder 

You can find this tutorial with images to facilitate the navigation on the [Google Docs file linked above](https://docs.google.com/document/d/1_USwRnq4MR6dlpCHX_2xd5flOwr3TpvJ9h362sOBOFg/edit).
```
