# inquire

## Installation

``python setup.py develop``
``pip install -e setup.py``

## Running Inquire

``python tests/icml22.py``

### To run the lunar lander domain

``python tests/icml22.py --domain lander``

### To run a "quick" lunar lander trial

``python tests/icml22.py --domain lander --queries 2 --runs 2 --tests 2 -M 20 -N 2``

### To visualize a lunar lander trajectory

``python visualize_lunar_lander_control.py </relative/path/to/file.csv>``
