# INQUIRE: INteractive Querying for User-aware Informative REasoning

The codebase for the corresponding paper (as titled).\
**Last README update:** June 24, 2022

## Abstract (abbreviated)

INQUIRE is the first Interactive Robot Learning algorithm to implement
(and optimize over) a generalized representation of information gain
across multiple interaction types. It can dynamically optimize its
interaction type (and respective optimal query) based on its current
learning status and the robot's state in the world, and users can bias
its selection of interaction types via customizable cost metrics.

[See the paper for more details](www.__.com)

## Paper citation

---

## Install

Note: This repo was built using **Python 3.10.2** and a **conda virtual environment**.

### If Conda ISN'T installed on your machine

Install Conda by following the [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
according to your machine's specifications.

### If Conda IS installed

From INQUIRE's top-level directory, run:

1. ``conda deactivate``
1. ``conda env create -n inquire --file environment.yml``
1. ``conda activate inquire``
1. ``pip install -e .``

### Install troubleshooting

#### Errors related to box2d

From INQUIRE's top-level directory, run:

1. ``conda install -c conda-forge swig``
1. ``pip install -e .``

---

## Run INQUIRE with...

### ...default settings

``python tests/corl22.py``

### ...a specific domain

``python tests/corl22.py --domain <environment name>``

e.g.

``python tests/corl22.py --domain lander``

### ...a script used to gather experimental data

From the top-level directory, run:

``bash run_inquire.sh``

### ...a quick LunarLander trial

#### (also useful when debugging the querying process)

``python tests/corl22.py --domain lander --queries 2 --runs 1 --tests 1``

### ...``python corl22.py --help`` for more command-line options

including experiment parameters, domain and agent specification,
and other algorithmic nuance.

---

## Visualize...

### ...an optimal LunarLander trajectory

``python tests/viz_weights.py --domain lander --weights "0.55 0.55, 0.41, 0.48" -I 1000``

### ...a saved LunarLander trajectory

``python tests/visualize_lunar_lander_control.py </relative/path/to/stored/trajectory.csv>``
