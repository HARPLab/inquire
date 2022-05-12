# inquire

## Installation

### via Conda

Create a conda virtual environment:

``conda create -n inquire python=3.8``

``conda deactivate``

``conda activate inquire``

Install Swig:

``conda install -c conda-forge swig``

From the top level of the Inquire directory, run:

``pip install -e .``

### Virtualenv Instructions

``sudo apt install python3.8 libpython3.8 libpython3.8-dev python3.8-venv swig``

``virtualenv -p python3.8 inquire_env``

``source inquire_env/bin/activate``

``pip install -e .``

## Running Inquire

``python tests/icml22.py``

### To run the lunar lander domain

``python tests/icml22.py --domain lander``

### To run a "quick" lunar lander trial

``python tests/icml22.py --domain lander --queries 2 --runs 1 --tests 1``

### The commandline arguments which led to 4 preference queries followed by one demo. query

``python tests/icml22.py --domain lander --runs 1 --queries 5 --tests 5 -M 5 -N 5 -I 50 --verbose``

### To visualize a lunar lander trajectory

``python visualize_lunar_lander_control.py </relative/path/to/file.csv>``

### To run binary feedback
``python tests/icml22.py -V -A bin-fb-only``

Note: at the moment this only implements binary feedback from the teacher side. From the agent side, it mocks the demo-only agent in the sense that it generates a single trajectory for its query. Additionally, the agent is not yet equipped to interpret binary feedback (in the form of +/- 1), meaning the script will crash after the first iteration. For now this is meant to only serve as a test from the teacher-side, with the agent side yet to be implemented.
