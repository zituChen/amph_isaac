# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import amph_isaac.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False



#---------------------------------------------------------------------------------
class DebugBaseContactForces(gym.Wrapper):
    """Print base_link contact force magnitude from ContactSensor."""

    def __init__(
        self,
        env,
        sensor_name: str = "contact_forces",
        robot_name: str = "robot",
        base_body_name: str = "base_link",
        threshold: float = 1.0,
        print_every: int = 1,
        only_when_done: bool = True,
    ):
        super().__init__(env)
        self.sensor_name = sensor_name
        self.robot_name = robot_name
        self.base_body_name = base_body_name
        self.threshold = threshold
        self.print_every = print_every
        self.only_when_done = only_when_done
        self._step_count = 0
        self._base_body_id = None  # resolved lazily

    def _resolve_ids(self):
        # Resolve body index once the sim is live
        if self._base_body_id is not None:
            return

        uenv = self.env.unwrapped
        robot = uenv.scene[self.robot_name]

        # IsaacLab articulations typically provide find_bodies(pattern or name).
        # We try common patterns to be robust.
        try:
            ids, _ = robot.find_bodies(self.base_body_name)
            self._base_body_id = int(ids[0])
            return
        except Exception:
            pass

        try:
            ids = robot.find_bodies(self.base_body_name)
            self._base_body_id = int(ids[0])
            return
        except Exception as e:
            raise RuntimeError(
                f"Could not resolve base body id for '{self.base_body_name}'. "
                f"Check the body/link name in USD/URDF."
            ) from e

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        if self._step_count % self.print_every != 0:
            return obs, rew, terminated, truncated, info

        self._resolve_ids()

        uenv = self.env.unwrapped
        sensor = uenv.scene.sensors[self.sensor_name]

        # net_forces_w_history: (num_envs, history_len, num_bodies, 3)
        forces_hist = sensor.data.net_forces_w_history
        # take most recent history frame
        forces = forces_hist[:, 0, self._base_body_id, :]  # (num_envs, 3)

        force_mag = torch.linalg.norm(forces, dim=-1)  # (num_envs,)

        # Determine which envs to print
        if self.only_when_done:
            done_mask = torch.as_tensor(terminated) | torch.as_tensor(truncated)
            idx = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        else:
            idx = torch.arange(force_mag.shape[0], device=force_mag.device)

        if idx.numel() > 0:
            fm = force_mag[idx].detach().cpu()
            fxyz = forces[idx].detach().cpu()

            # Print a short summary + first few envs
            worst = float(fm.max().item())
            mean = float(fm.mean().item())
            print(
                f"[DEBUG][step={self._step_count}] base_link contact |F|: "
                f"mean={mean:.3f}  max={worst:.3f}  (threshold={self.threshold})"
            )
            for k in range(min(5, idx.numel())):
                env_id = int(idx[k].item())
                print(f"  env {env_id}: |F|={fm[k]:.3f}  F={tuple(fxyz[k].tolist())}")

        return obs, rew, terminated, truncated, info
#----------------------------------------------------------------------------------------------


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ------------------- DEBUG JOINT ORDER -------------------
    # uenv = env.unwrapped
    # robot = uenv.scene["robot"]

    # 1) The simulator/articulation joint order (index -> joint name)
    # IsaacLab commonly exposes one of these attributes depending on version.
    # joint_names = None
    # for attr in ["joint_names", "dof_names"]:
    #     if hasattr(robot, attr):
    #         joint_names = list(getattr(robot, attr))
    #         break

    # if joint_names is None and hasattr(robot, "data") and hasattr(robot.data, "joint_names"):
    #     joint_names = list(robot.data.joint_names)

    # print("\n[DEBUG] Robot joint order (index -> name):")
    # if joint_names is not None:
    #     for i, n in enumerate(joint_names):
    #         print(f"  {i:02d}: {n}")
    # else:
    #     print("  Could not find joint names on robot. Try robot, robot.data attributes.")

    # # 2) Compare against cfg's joint_sdk_names (if present)
    # sdk_list = getattr(env_cfg.scene.robot, "joint_sdk_names", None)
    # if sdk_list is not None:
    #     print("\n[DEBUG] cfg.scene.robot.joint_sdk_names:")
    #     for i, n in enumerate(list(sdk_list)):
    #         print(f"  sdk[{i:02d}]: {n}")

    #     if joint_names is not None:
    #         same = (list(sdk_list) == joint_names)
    #         print(f"\n[DEBUG] Exact match (sdk_list == sim_joint_order)? {same}")

    #         # Show first mismatch if any
    #         if not same:
    #             min_len = min(len(sdk_list), len(joint_names))
    #             for i in range(min_len):
    #                 if sdk_list[i] != joint_names[i]:
    #                     print(f"[DEBUG] First mismatch at index {i}: sdk='{sdk_list[i]}' vs sim='{joint_names[i]}'")
    #                     break
    #             if len(sdk_list) != len(joint_names):
    #                 print(f"[DEBUG] Length differs: sdk={len(sdk_list)} vs sim={len(joint_names)}")
    # ---------------------------------------------------------

    # ------------------- DEBUG ACTION TERM JOINT IDS -------------------
    # if hasattr(uenv, "action_manager"):
    #     am = uenv.action_manager

    #     # Try to find the joint position action term
    #     term = None
    #     for attr in ["terms", "_terms"]:
    #         if hasattr(am, attr):
    #             for k, v in getattr(am, attr).items():
    #                 if "joint" in k:
    #                     term = v
    #                     print(f"[DEBUG] Found action term: {k}")
    #                     break

    #     if term is None:
    #         print("[DEBUG] Could not find joint action term")
    #     else:
    #         # Try common joint-id attributes
    #         ids = None
    #         for attr in ["joint_ids", "_joint_ids", "dof_ids", "_dof_ids"]:
    #             if hasattr(term, attr):
    #                 ids = getattr(term, attr)
    #                 print(f"[DEBUG] term.{attr}: {ids}")
    #                 break

    #         if ids is None:
    #             print("[DEBUG] No joint id information found on action term")
    #         else:
    #             # -------- FIX: handle slice(None) --------
    #             if isinstance(ids, slice):
    #                 # slice(None) => all joints, in articulation order
    #                 ids_list = list(range(len(joint_names)))[ids]
    #             else:
    #                 # tensor / list / numpy array
    #                 ids_list = [int(i) for i in list(ids)]

    #             print("\n[DEBUG] Action index -> joint name mapping:")
    #             for a_i, j_i in enumerate(ids_list):
    #                 print(f"  a[{a_i:02d}] -> joint[{j_i:02d}] = {joint_names[j_i]}")
    # -------------------------------------------------------------------

    
    # # --- DEBUG: print base_link contact forces ---
    # env = DebugBaseContactForces(
    #     env,
    #     sensor_name="contact_forces",
    #     robot_name="robot",
    #     base_body_name="base_link",
    #     threshold=1.0,
    #     print_every=10,          # increase to e.g. 10/50 to reduce spam
    #     only_when_done=True,    # set False to print every step for all envs
    # )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
