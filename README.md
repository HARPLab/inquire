# inquire

## Installation

### via Conda

From the top level of the Inquire directory, run:

1. ``conda deactivate``
1. ``conda env create -n inquire --file environment.yml``
1. ``conda activate inquire``
1. ``pip install -e .``

If you encounter errors related to ``box2d``, run the following additional commands
from the top-level directory:

1. ``conda install -c conda-forge swig``
1. ``pip install -e .``

### via Virtualenv

#### NOTE these commands are outdated as of 5/30/22. See conda installation

``sudo apt install python3.8 libpython3.8 libpython3.8-dev python3.8-venv swig``

``virtualenv -p python3.8 inquire_env``

``source inquire_env/bin/activate``

``pip install -e .``

## To run Inquire with default settings

``python tests/icml22.py``

### To run the lunar lander domain

``python tests/icml22.py --domain lander``

### To run a "quick" lunar lander trial

``python tests/icml22.py --domain lander --queries 2 --runs 1 --tests 1``

### To visualize a lunar lander trajectory

``python visualize_lunar_lander_control.py </relative/path/to/trajectory.csv>``

### To visualize an optimal lunar lander trajectory

``python tests/viz_weights.py --domain lander --weights "0.55 0.55, 0.41, 0.48" -I 1000``

### Debugging the querying process with lunar lander

Commandline arguments that led to 4 preference queries then one demo. query:

```bash
python tests/icml22.py --domain lander --runs 1 --queries 5 --tests 5 -M 5 -N 5 -I 50 -V
```

### To run binary feedback

``python tests/icml22.py -V -A bin-fb-only``

Note: at the moment this only implements binary feedback from the teacher side.
From the agent side, it mocks the demo-only agent in the sense that it generates
a single trajectory for its query. Additionally, the agent is not yet equipped to
interpret binary feedback (in the form of +/- 1), meaning the script will crash
after the first iteration. For now this is meant to only serve as a test from the
teacher-side, with the agent side yet to be implemented.

### To instantiate a DemPref agent

1. Designate DemPref-specific parameters within the ``set_agent_config.py`` file
1. From within the ``inquire/agents/`` sub-directory, run:

   ```bash
   python set_agent_config.py
   ```

   A new .csv file should now be in the ``agents/`` sub-directory.
1. From the ``inquire/tests/`` sub-directory, run:

   ```bash
   python icml22.py --agent dempref --domain lander
   ```
