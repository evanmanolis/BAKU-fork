import argparse
import os
import pickle as pkl
from pathlib import Path

import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from sentence_transformers import SentenceTransformer


DEFAULT_DATASET_PATH = Path(os.environ.get("LIBERO_DATASET_PATH", "/path/to/datasets"))
DEFAULT_SAVE_DATA_PATH = Path(
    os.environ.get("BAKU_LIBERO_EXPERT_PATH", "../../expert_demos/libero")
)
DEMO_SUFFIX = "_demo.hdf5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw LIBERO HDF5 demonstrations into BAKU pickle demos."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Directory containing raw LIBERO suite folders such as libero_90/.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=DEFAULT_SAVE_DATA_PATH,
        help="Directory where converted BAKU pickle files will be written.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["libero_10", "libero_90"],
        help="LIBERO benchmark folders to convert.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Square render size for converted pixel observations.",
    )
    parser.add_argument(
        "--max-tasks-per-benchmark",
        type=int,
        default=None,
        help="Optional smoke-test limit per benchmark.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reconvert tasks even when the output pickle already exists.",
    )
    return parser.parse_args()


def task_name_from_demo(task_file):
    name = task_file.name
    if not name.endswith(DEMO_SUFFIX):
        raise ValueError(f"unexpected LIBERO demo filename: {task_file}")
    return name[: -len(DEMO_SUFFIX)]


def render_demo_file(task_file, task_suite, lang_model, img_size):
    with h5py.File(task_file, "r") as handle:
        data = handle["data"]
        task_name = task_name_from_demo(task_file)
        task_id = task_suite.get_task_names().index(task_name)
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=img_size,
            camera_widths=img_size,
        )
        env.reset()

        observations = []
        states = []
        actions = []
        rewards = []

        for demo in data.keys():
            print(f"  Processing {demo}", flush=True)
            demo_data = data[demo]

            observation = {
                "robot_states": np.array(demo_data["robot_states"], dtype=np.float32)
            }

            pixels, pixels_ego = [], []
            joint_states, eef_states, gripper_states = [], [], []
            for i in range(len(demo_data["states"])):
                obs = env.regenerate_obs_from_state(demo_data["states"][i])
                pixels.append(obs["agentview_image"][::-1])
                pixels_ego.append(obs["robot0_eye_in_hand_image"][::-1])
                joint_states.append(obs["robot0_joint_pos"])
                eef_states.append(
                    np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                )
                gripper_states.append(obs["robot0_gripper_qpos"])

            observation["pixels"] = np.array(pixels, dtype=np.uint8)
            observation["pixels_egocentric"] = np.array(pixels_ego, dtype=np.uint8)
            observation["joint_states"] = np.array(joint_states, dtype=np.float32)
            observation["eef_states"] = np.array(eef_states, dtype=np.float32)
            observation["gripper_states"] = np.array(gripper_states, dtype=np.float32)

            observations.append(observation)
            states.append(np.array(demo_data["states"], dtype=np.float32))
            actions.append(np.array(demo_data["actions"], dtype=np.float32))
            rewards.append(np.array(demo_data["rewards"], dtype=np.float32))

        return {
            "observations": observations,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "task_emb": lang_model.encode(env.language_instruction),
        }


def main():
    args = parse_args()
    args.save_path.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    task_files = []
    for suite_name in args.benchmarks:
        benchmark_path = args.dataset_path / suite_name
        files = sorted(benchmark_path.glob("*.hdf5"))
        if args.max_tasks_per_benchmark is not None:
            files = files[: args.max_tasks_per_benchmark]
        task_files.extend((suite_name, task_file) for task_file in files)

    if not task_files:
        raise FileNotFoundError(
            f"no LIBERO HDF5 files found under {args.dataset_path} for {args.benchmarks}"
        )

    tasks_stored = 0
    for suite_name in args.benchmarks:
        print(f"############################# {suite_name} #############################")
        suite_files = [
            task_file for suite, task_file in task_files if suite == suite_name
        ]
        if not suite_files:
            print(f"No files found for {suite_name}; skipping.")
            continue

        save_benchmark_path = args.save_path / suite_name
        save_benchmark_path.mkdir(parents=True, exist_ok=True)
        task_suite = benchmark_dict[suite_name]()

        for task_file in suite_files:
            save_data_path = (
                save_benchmark_path / f"{task_name_from_demo(task_file)}.pkl"
            )
            if save_data_path.exists() and not args.overwrite:
                print(f"Skipping existing {save_data_path}")
                tasks_stored += 1
                continue

            print(
                f"Processing {tasks_stored + 1}/{len(task_files)}: {task_file}",
                flush=True,
            )
            converted = render_demo_file(
                task_file, task_suite, lang_model, args.img_size
            )
            with save_data_path.open("wb") as handle:
                pkl.dump(converted, handle)
            print(f"Saved to {save_data_path}", flush=True)
            tasks_stored += 1


if __name__ == "__main__":
    main()
