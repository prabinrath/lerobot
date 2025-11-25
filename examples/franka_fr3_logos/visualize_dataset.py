#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Copyright (c) 2025 Prabin Kumar Rath
# Co-developed with Claude Sonnet 4.5 (Anthropic)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

"""
Visualize any LeRobot dataset using the Rerun visualizer.

This script loads any LeRobot dataset and displays it using the Rerun visualizer,
showing camera feeds, robot states, actions, and commanded positions for specified episodes.

Usage:
    # Visualize episode 0 from a local dataset
    python visualize_dataset.py --repo-id franka_fr3_softtoy_in_basket --root ./datasets/franka_fr3_softtoy_in_basket
    
    # Visualize specific episode
    python visualize_dataset.py --repo-id my_dataset --root ./datasets/my_dataset --episode-index 5
    
    # Visualize multiple episodes
    python visualize_dataset.py --repo-id my_dataset --root ./datasets/my_dataset --episode-index 0 5 10
    
    # Save visualization to file
    python visualize_dataset.py --repo-id my_dataset --root ./datasets/my_dataset --save --output-dir ./viz_output
    
    # Show dataset information
    python visualize_dataset.py --repo-id my_dataset --root ./datasets/my_dataset --info
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 tensor to HWC uint8 numpy array for visualization."""
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    """Visualize dataset including observation.command data for commanded positions."""
    if save:
        assert output_dir is not None, "Set output directory with `--output-dir path/to/directory`."

    repo_id = dataset.repo_id

    logger.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logger.info("Starting Rerun")
    if mode not in ["local", "distant"]:
        raise ValueError(f"Invalid mode: {mode}")

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)
    gc.collect()

    if mode == "distant":
        rr.serve_web_viewer(open_browser=False, web_port=web_port)

    logger.info("Logging to Rerun")
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["frame_index"][i].item())
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            # Display camera images
            for key in dataset.meta.camera_keys:
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # Display action values
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rr.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

            # Display observed state
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

            # Display commanded positions
            if "observation.command" in batch:
                for dim_idx, val in enumerate(batch["observation.command"][i]):
                    rr.log(f"command/{dim_idx}", rr.Scalars(val.item()))

            # Display other metrics
            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))
            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))
            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

    if mode == "local" and save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path
    elif mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")
    
    return None


def visualize_episode(
    repo_id: str,
    dataset_root: str | None,
    episode_index: int,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 32,
    mode: str = "local",
) -> None:
    """Visualize a single episode of the dataset."""
    logger.info(f"Loading dataset: {repo_id}")
    if dataset_root:
        logger.info(f"Dataset root: {dataset_root}")
    else:
        logger.info("Loading from Hugging Face hub")
    logger.info(f"Visualizing episode {episode_index}")
    
    # Load the dataset for the specific episode
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, episodes=[episode_index])
    
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Total frames in episode {episode_index}: {len(dataset)}")
    logger.info(f"  Robot type: {dataset.meta.robot_type}")
    logger.info(f"  FPS: {dataset.meta.fps}")
    logger.info(f"  Camera keys: {dataset.meta.camera_keys}")
    logger.info(f"  Total episodes: {dataset.meta.total_episodes}")
    
    # Visualize the dataset
    output_path = visualize_dataset(
        dataset=dataset,
        episode_index=episode_index,
        batch_size=batch_size,
        mode=mode,
        save=save,
        output_dir=Path(output_dir) if output_dir else None,
    )
    
    if save and output_path:
        logger.info(f"Visualization saved to: {output_path}")
        logger.info(f"To view: rerun {output_path}")
    else:
        logger.info("Visualization opened in Rerun viewer")


def visualize_multiple_episodes(
    repo_id: str,
    dataset_root: str | None,
    episode_indices: list[int],
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 32,
    delay_between_episodes: float = 2.0,
) -> None:
    """Visualize multiple episodes in sequence."""
    logger.info(f"Visualizing {len(episode_indices)} episodes: {episode_indices}")
    
    for i, episode_idx in enumerate(episode_indices):
        logger.info(f"Visualizing episode {episode_idx} ({i+1}/{len(episode_indices)})")
        
        try:
            visualize_episode(
                repo_id=repo_id,
                dataset_root=dataset_root,
                episode_index=episode_idx,
                save=save,
                output_dir=output_dir,
                batch_size=batch_size,
            )
            
            if i < len(episode_indices) - 1 and not save:
                logger.info(f"Waiting {delay_between_episodes}s before next episode...")
                time.sleep(delay_between_episodes)
        except Exception as e:
            logger.error(f"Failed to visualize episode {episode_idx}: {e}")
            continue


def get_dataset_info(repo_id: str, dataset_root: str | None) -> None:
    """Print information about the dataset."""
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, episodes=[0])
    
    print(f"\n=== Dataset Information ===")
    print(f"Repository ID: {repo_id}")
    print(f"Robot type: {dataset.meta.robot_type}")
    print(f"Total episodes: {dataset.meta.total_episodes}")
    print(f"Total frames: {dataset.meta.total_frames}")
    print(f"FPS: {dataset.meta.fps}")
    print(f"Camera keys: {dataset.meta.camera_keys}")
    print(f"Video keys: {dataset.meta.video_keys}")
    
    print(f"\n=== Features ===")
    for feature_name, feature_info in dataset.meta.features.items():
        print(f"  {feature_name}:")
        print(f"    dtype: {feature_info['dtype']}")
        print(f"    shape: {feature_info['shape']}")
        if 'names' in feature_info and feature_info['names']:
            print(f"    names: {feature_info['names']}")
    
    print(f"\n=== Episode Range ===")
    print(f"Available episodes: 0 to {dataset.meta.total_episodes - 1}")
    
    if hasattr(dataset.meta, 'tasks') and dataset.meta.tasks is not None:
        print(f"\n=== Tasks ===")
        print(dataset.meta.tasks)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LeRobot dataset with command data support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--repo-id", type=str, required=True,
                       help="Repository ID (e.g., 'lerobot/pusht' or 'my_dataset')")
    parser.add_argument("--episode-index", type=int, nargs="*", default=[0],
                       help="Episode index(es) to visualize. Default: [0]")
    parser.add_argument("--all-episodes", action="store_true",
                       help="Visualize all episodes")
    parser.add_argument("--root", type=str, default=None,
                       help="Root directory for local datasets")
    parser.add_argument("--save", action="store_true",
                       help="Save visualization to .rrd file")
    parser.add_argument("--output-dir", type=str, default="./visualization_outputs",
                       help="Output directory for saved visualizations. Default: ./visualization_outputs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for data loading. Default: 32")
    parser.add_argument("--mode", type=str, choices=["local", "distant"], default="local",
                       help="Visualization mode. Default: local")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between episodes when visualizing multiple. Default: 2.0")
    parser.add_argument("--info", action="store_true",
                       help="Show dataset information and exit")
    
    args = parser.parse_args()
    
    # Validate dataset path
    if args.root and not Path(args.root).exists():
        logger.error(f"Dataset path does not exist: {args.root}")
        return
    
    # Show info and exit
    if args.info:
        get_dataset_info(args.repo_id, args.root)
        return
    
    # Determine episodes to visualize
    if args.all_episodes:
        temp_dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, episodes=[0])
        episode_indices = list(range(temp_dataset.meta.total_episodes))
        logger.warning(f"Visualizing ALL {len(episode_indices)} episodes. This may take a while!")
    else:
        episode_indices = args.episode_index
    
    # Validate episode indices
    temp_dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, episodes=[0])
    max_episode = temp_dataset.meta.total_episodes - 1
    for episode_idx in episode_indices:
        if episode_idx < 0 or episode_idx > max_episode:
            logger.error(f"Episode index {episode_idx} is out of range [0, {max_episode}]")
            return
    
    # Create output directory if saving
    if args.save:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizations will be saved to: {args.output_dir}")
    
    # Visualize episodes
    if len(episode_indices) == 1:
        visualize_episode(
            repo_id=args.repo_id,
            dataset_root=args.root,
            episode_index=episode_indices[0],
            save=args.save,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            mode=args.mode,
        )
    else:
        visualize_multiple_episodes(
            repo_id=args.repo_id,
            dataset_root=args.root,
            episode_indices=episode_indices,
            save=args.save,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            delay_between_episodes=args.delay,
        )


if __name__ == "__main__":
    main()