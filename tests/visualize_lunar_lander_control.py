import numpy as np
import time
import gym
import argparse
import pandas as pd

ll_env = gym.make("LunarLanderContinuous-v2")


def get_data(file_name: str):
    df = pd.read_csv(file_name)
    data = df.to_numpy()[:, 1:]
    controls = data[:, 0:2]
    seed = df["state seed"][0]

    return data, controls, seed


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        default=None,
        help="The relative path to the .csv file with the controls you'd like to visualize.",
    )

    return parser.parse_args()


def watch(env, controls, start_seed: int = None, frame_delay_ms: int = 20):
    env.seed(int(start_seed))
    env.reset()
    for i in range(controls.shape[0]):
        env.render()
        a = controls[i, :]
        observation, reward, done, info = env.step(a)
        time.sleep(frame_delay_ms / 1000)
        if done:
            break
    env.close()


def main():
    """Main method."""
    user_input = parse_input()

    try:
        assert user_input.file is not None
        data, controls_from_file, seed_from_file = get_data(user_input.file)

        watch(
            env=ll_env, controls=controls_from_file, start_seed=seed_from_file
        )
    except AssertionError:
        print("User must include the path to the .csv file as an argument.")
        return


if __name__ == "__main__":
    main()
