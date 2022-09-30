import time

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from inquire.utils.args_handler import ArgsHandler
from inquire.run import *

def save_data(
    data: Union[list, np.ndarray],
    labels: list,
    num_runs: int,
    directory: str,
    filename: str,
    subdirectory: str = None,
) -> None:
    """Save data to file in directory."""
    agents = labels
    data_stack = np.stack(data, axis=1)
    tasks = [i for i in range(data_stack.shape[0])]
    runs = [i for i in range(num_runs)]
    test_states = [i for i in range(data_stack.shape[3])]
    index = pd.MultiIndex.from_product(
        [tasks, agents, runs, test_states],
        names=["task", "agent", "run", "test_state"],
    )
    if subdirectory != None:
        path = Path(directory) / Path(subdirectory)
    else:
        path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True)
    df = pd.DataFrame(
        data_stack.reshape(-1, data_stack.shape[-1]), index=index
    )
    final_path = path / Path(filename)
    df.to_csv(final_path)
    print(f"Data saved to {final_path}")
    return df

if __name__ == "__main__":
    args = ArgsHandler()
    domain = args.setup_domain()
    teacher = args.setup_teacher()
    agents, agent_names = args.setup_agents()

    ## Run evaluation ##
    data = {}
    data["distance"] = []
    data["performance"] = []
    data["query_types"] = []
    start = time.perf_counter()
    eval_start_time = time.strftime("_%m:%d:%H:%M", time.localtime())
    for agent, name in zip(agents, agent_names):
        print("Evaluating " + name + " agent...                    ")
        perf, dist, q_type = run(
            domain,
            teacher,
            agent,
            args.num_tasks,
            args.num_runs,
            args.num_queries,
            args.num_test_states,
            args.step_size,
            args.conv_threshold,
            use_cached_trajectories=args.use_cache,
            static_state=args.static_state,
            verbose=args.verbose,
            reuse_weights=args.reuse_weights
        )
        if args.output_name is not None:
            dist_sum = np.sum(dist)
            perf_sum = np.sum(perf)
            with open(args.output_dir + "/" + "overview.txt", "a+") as f:
                f.write(
                    args.output_name
                    + ", "
                    + str(dist_sum)
                    + ", "
                    + str(perf_sum)
                    + "\n"
                )
        data["distance"].append(dist)
        data["performance"].append(perf)
        data["query_types"].append(q_type)
    elapsed = time.perf_counter() - start
    if args.verbose:
        print(f"The complete evaluation took {elapsed:.4} seconds.")
    if args.output_name is None:
        name = domain.__class__.__name__ + eval_start_time
    else:
        name = args.output_name
    for d in list(data.keys()):
        save_data(
            data=data[d],
            labels=agent_names,
            num_runs=args.num_runs,
            directory=args.output_dir,
            filename=name + f"_{d}.csv",
            subdirectory=domain.__class__.__name__,
        )

