# inquire

## Installation

``python setup.py develop``

(The alleged 'safer' method):

``pip install -e .``

## Running Inquire

``python tests/icml22.py``

### To run the lunar lander domain

``python tests/icml22.py --domain lander``

### To run a "quick" lunar lander trial

``python tests/icml22.py --domain lander --queries 2 --runs 2 --tests 2 -M 20 -N 2``

### The commandline arguments which led to 4 preference queries followed by one demo. query

``python tests/icml22.py --domain lander --runs 1 --queries 5 --tests 5 -M 5 -N 5 -I 50 --verbose``

### To visualize a lunar lander trajectory

``python visualize_lunar_lander_control.py </relative/path/to/file.csv>``
