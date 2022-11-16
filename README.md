# INQUIRE: INteractive Querying for User-aware Informative REasoning

The codebase for the corresponding paper (as titled).

## Abstract (abbreviated)

INQUIRE is the first Interactive Robot Learning algorithm to implement
(and optimize over) a generalized representation of information gain
across multiple interaction types. It can dynamically optimize its
interaction type (and respective optimal query) based on its current
learning status and the robot's state in the world, and users can bias
its selection of interaction types via customizable cost metrics.

[See the paper for more details.](https://openreview.net/pdf?id=3CQ3Vt0v99)

## Paper citation
```
Fitzgerald, T., Koppol, P., Callaghan, P., Wong, R., Kroemer, O., Simmons, R., Admoni, H. “INQUIRE: INteractive Querying for User-Aware Informative REasoning”. In Sixth Conference on Robot Learning (CoRL). PMLR, 2022.
```

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

``python scripts/corl22.py``

### ...a specific domain

``python scripts/corl22.py --domain <environment name>``

e.g.

``python scripts/corl22.py --domain lander``

### ...the script used to gather experimental data for the CoRL'22 paper

From the top-level directory, run:

``bash scripts/run_inquire.sh lander``

Substitute for other domains such as ``linear_combo``, ``linear_system``, or ``pizza``. 

**Note:** this script assumes you have already run the [cache script](#cache-trajectories).

### ...a quick LunarLander trial

#### (also useful when debugging the querying process)

``python scripts/corl22.py --domain lander --queries 2 --runs 1 --tests 1``

### ...``python scripts/corl22.py --help`` for more command-line options

including experiment parameters, domain and agent specification,
and other algorithmic nuance.

---

## Visualize...

### ...an optimal LunarLander trajectory

``python viz/viz_weights.py --domain lander --weights "0.55 0.55, 0.41, 0.48" -I 1000``

### ...a saved LunarLander trajectory

``python viz/visualize_lunar_lander_control.py </relative/path/to/stored/trajectory.csv>``

---

## Cache Trajectories

For faster testing, you can cache a set of trajectory samples ahead of time. For example:

``python scripts/cache_trajectories.py --domain lander -Z 1 -X 250 -N 1000``

This will cache 1000 trajectories for each of 250 states for 1 task (i.e., 1 ground-truth weight vector). 
Make sure that X >= [(number of runs * number of queries) + number of test states] and that N >= the number of trajectory samples used during evaluations.

You can then use this cache by adding the ``--use_cache`` command-line argument.
