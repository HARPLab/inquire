# inquire

Last README update: June 8, 2022

Note: Inquire uses **Python 3.10.2**

## Install via Conda

From Inquire's top-level directory, run:

1. ``conda deactivate``
1. ``conda env create -n inquire --file environment.yml``
1. ``conda activate inquire``
1. ``pip install -e .``

### Conda install troubleshooting

#### Errors related to box2d

From Inquire's top-level directory, run:

1. ``conda install -c conda-forge swig``
1. ``pip install -e .``

---

## Install via Virtualenv

### NOTE: These commands are outdated as of 5/30/22. See conda installation

``sudo apt install python3.8 libpython3.8 libpython3.8-dev python3.8-venv swig``

``virtualenv -p python3.8 inquire_env``

``source inquire_env/bin/activate``

``pip install -e .``

---

## Run Inquire with...

### ...default settings

``python tests/corl22.py``

### ...a specific domain

``python tests/corl22.py --domain <environment name>``

e.g.

``python tests/corl22.py --domain lander``

### ...a quick lunar lander trial

#### also useful when debugging the querying process

``python tests/corl22.py --domain lander --queries 2 --runs 1 --tests 1``

---

## Visualize ...

### ...an optimal lunar lander trajectory

``python tests/viz_weights.py --domain lander --weights "0.55 0.55, 0.41, 0.48" -I 1000``

### ...a saved lunar lander trajectory

``python visualize_lunar_lander_control.py </relative/path/to/stored/trajectory.csv>``

---
### To run binary feedback

``python tests/corl22.py -V -A bin-fb-only``

Note: at the moment this only implements binary feedback from the teacher side.
From the agent side, it mocks the demo-only agent in the sense that it generates
a single trajectory for its query. Additionally, the agent is not yet equipped to
interpret binary feedback (in the form of +/- 1), meaning the script will crash
after the first iteration. For now this is meant to only serve as a test from the
teacher-side, with the agent side yet to be implemented.
