"""Create a .csv to be read by external agents."""
import argparse
import time
from pathlib import Path

import pandas as pd

# Most parameters assigned according to arguments in
# dempref/experiments/main_experiment/main_experiment.py:
dempref_agent_params = {
    "domain": "lander",
    "teacher_type": "opt",
    "update_func": "approx",
    "epsilon": 0.0,
    "beta_demo": 0.1,
    "beta_pref": 5,
    "beta_teacher": 1,
    "n_demos": 3,
    "n_iters_exp": 8,
    "n_pref_iters": 25,
    "n_samples_exp": 50000,
    "n_samples_summ": 50000,
    "query_option_count": 2,
    "opt_iter_count": 50,
    "trajectory_length": 10,
    "trim_start": 0,
    "gen_demos": True,
    "gen_scenario": False,
    "incl_prev_query": False,
    "true_weight": [[-0.4, 0.4, -0.2, -0.7]],
}
agents = {"dempref": dempref_agent_params}

# Get I/O commandline arguments:
parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent", type=str, default="dempref", choices=["dempref"]
)
parser.add_argument("--path", type=str, default=".")
parser.add_argument("--suffix", type=str, default="_agent.csv")
parser.add_argument("--filename", type=str, default="")


def create_csv() -> None:
    """Use global *_agent_params to create agent's parameter .csv."""
    args = parser.parse_args()

    path = args.path
    if args.filename != "":
        file = args.filename + args.suffix
    else:
        time_prefix = time.strftime("%m:%d:%H:%M_", time.localtime())
        file = time_prefix + args.agent + args.suffix
    if path == ".":
        path = Path.cwd()
    else:
        path = Path(path)

    if not path.exists():
        path.mkdir(parents=True)

    full_path = path / Path(file)
    df = pd.DataFrame(agents[args.agent])
    df.to_csv(full_path)


def main() -> None:
    """Run the program."""
    create_csv()


if __name__ == "__main__":
    main()
