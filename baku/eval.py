#!/usr/bin/env python3

import warnings
import os
import gc
from types import SimpleNamespace

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from libero.libero import benchmark

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    try:
        torch.backends.cuda.preferred_linalg_library("magma")
    except RuntimeError:
        pass


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


def close_envs(envs):
    for env in envs:
        close = getattr(env, "close", None)
        if close is not None:
            close()


class TextOnlyEvalDataset:
    def __init__(self, cfg, stats):
        self.stats = stats
        self.envs_till_idx = 0

        tasks_by_scene = {
            task_name: scene[task_name]
            for scene in cfg.suite.task.tasks
            for task_name in scene
        }
        task_names = []
        for scene in cfg.suite.task.scenes:
            task_names.extend(tasks_by_scene[scene])

        task_suite = benchmark.get_benchmark_dict()[cfg.suite.task.suite]()
        suite_names = task_suite.get_task_names()
        task_languages = []
        for task_name in task_names:
            task = task_suite.get_task(suite_names.index(task_name))
            task_languages.append(task.language)

        lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.task_embs = lang_model.encode(task_languages)
        self.envs_till_idx = len(self.task_embs)

    def sample_test(self, env_idx, step=None):
        return {
            "prompt_pixels": None,
            "prompt_pixels_egocentric": None,
            "prompt_proprioceptive": None,
            "prompt_actions": None,
            "task_emb": self.task_embs[env_idx],
        }


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        bc_payload = None
        if cfg.get("text_only_eval", False):
            if cfg.prompt != "text":
                raise ValueError("text_only_eval requires prompt=text")
            if cfg.bc_weight is None:
                raise ValueError("text_only_eval requires bc_weight")
            bc_snapshot = Path(cfg.bc_weight)
            if not bc_snapshot.exists():
                raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
            with bc_snapshot.open("rb") as f:
                bc_payload = torch.load(f, map_location="cpu", weights_only=False)
            dataset = TextOnlyEvalDataset(cfg, bc_payload["stats"])
            self.expert_replay_loader = SimpleNamespace(dataset=dataset)
            self.expert_replay_iter = None
        else:
            # load data
            dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
            self.expert_replay_loader = make_expert_replay_loader(
                dataset_iterable, self.cfg.batch_size
            )
            self.expert_replay_iter = iter(self.expert_replay_loader)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        if cfg.get("text_only_eval", False):
            self.cfg.suite.task_make_fn.max_episode_len = bc_payload["max_episode_len"]
            self.cfg.suite.task_make_fn.max_state_dim = bc_payload.get(
                "max_state_dim", cfg.text_only_max_state_dim
            )
        else:
            self.cfg.suite.task_make_fn.max_episode_len = (
                self.expert_replay_loader.dataset._max_episode_len
            )
            self.cfg.suite.task_make_fn.max_state_dim = (
                self.expert_replay_loader.dataset._max_state_dim
            )
        if self.cfg.suite.name == "dmc":
            self.cfg.suite.task_make_fn.max_action_dim = (
                self.expert_replay_loader.dataset._max_action_dim
            )
        # Create a single probe env for specs before all eval render contexts exist.
        original_eval = self.cfg.suite.task_make_fn.eval
        self.cfg.suite.task_make_fn.eval = False
        probe_env, _ = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.agent = make_agent(
            probe_env[0].observation_spec(), probe_env[0].action_spec(), cfg
        )
        close_envs(probe_env)
        del probe_env
        gc.collect()
        torch.cuda.empty_cache()

        self._snapshot_loaded = False
        if cfg.bc_weight is not None:
            bc_snapshot = Path(cfg.bc_weight)
            if not bc_snapshot.exists():
                raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
            print(f"loading bc weight: {bc_snapshot}")
            self.load_snapshot({"bc": bc_snapshot})
            self._snapshot_loaded = True

        self.cfg.suite.task_make_fn.eval = original_eval
        self.env = []
        self.task_descriptions = []

        self.envs_till_idx = self.expert_replay_loader.dataset.envs_till_idx
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        if not cfg.get("text_only_eval", False):
            self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    def make_eval_env(self, env_idx):
        original_only_env_idx = self.cfg.suite.task_make_fn.only_env_idx
        self.cfg.suite.task_make_fn.only_env_idx = env_idx
        try:
            env, task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)
        finally:
            self.cfg.suite.task_make_fn.only_env_idx = original_only_env_idx
        return env[0], task_descriptions

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def eval(self):
        self.agent.train(False)
        episode_rewards = []
        successes = []
        for env_idx in range(self.envs_till_idx):
            print(f"evaluating env {env_idx}")
            env = None
            try:
                env, _ = self.make_eval_env(env_idx)
                episode, total_reward = 0, 0
                eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
                success = []

                while eval_until_episode(episode):
                    time_step = env.reset()
                    self.agent.buffer_reset()
                    step = 0

                    # prompt
                    if self.cfg.prompt != None and self.cfg.prompt != "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(env_idx)
                    else:
                        prompt = None

                    if episode == 0:
                        self.video_recorder.init(env, enabled=True)

                    # plot obs with cv2
                    while not time_step.last():
                        if self.cfg.prompt == "intermediate_goal":
                            prompt = self.expert_replay_loader.dataset.sample_test(
                                env_idx, step
                            )
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            action = self.agent.act(
                                time_step.observation,
                                prompt,
                                self.expert_replay_loader.dataset.stats,
                                step,
                                self.global_step,
                                eval_mode=True,
                            )
                        time_step = env.step(action)
                        self.video_recorder.record(env)
                        total_reward += time_step.reward
                        step += 1

                        if self.cfg.suite.name == "calvin" and time_step.reward == 1:
                            self.agent.buffer_reset()

                    episode += 1
                    success.append(time_step.observation["goal_achieved"])
                self.video_recorder.save(f"{self.global_frame}_env{env_idx}.mp4")
                episode_rewards.append(total_reward / episode)
                successes.append(np.mean(success))
            finally:
                if env is not None:
                    close_envs([env])
                    del env
                gc.collect()
                torch.cuda.empty_cache()

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[: self.envs_till_idx]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f, weights_only=False)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f, weights_only=False)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload, eval=True)


@hydra.main(config_path="cfgs", config_name="config_eval")
def main(cfg):
    from eval import WorkspaceIL as W

    root_dir = Path.cwd()
    workspace = W(cfg)

    # Load weights
    if not workspace._snapshot_loaded:
        snapshots = {}
        # bc
        bc_snapshot = Path(cfg.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        print(f"loading bc weight: {bc_snapshot}")
        snapshots["bc"] = bc_snapshot
        workspace.load_snapshot(snapshots)

    workspace.eval()


if __name__ == "__main__":
    main()
