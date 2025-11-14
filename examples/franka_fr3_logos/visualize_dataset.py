#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Visualize any LeRobot dataset using the LeRobot visualizer.

This script loads any LeRobot dataset and displays it using the Rerun visualizer,
showing camera feeds, robot states, and actions for specified episodes.

Usage:
    # Visualize episode 0 from a dataset in the current directory
    python visualize_dataset.py --repo-id my_dataset
    
    # Visualize from a specific path
    python visualize_dataset.py --repo-id my_dataset --root ./datasets/my_dataset
    
    # Visualize specific episodes
    python visualize_dataset.py --repo-id my_dataset --episode-index 5
    
    # Save visualization to file
    python visualize_dataset.py --repo-id my_dataset --episode-index 0 --save --output-dir ./viz_output
    
    # Visualize multiple episodes
    python visualize_dataset.py --repo-id my_dataset --episode-index 0 5 10

Examples:
    # Basic visualization of soft toy dataset
    python visualize_dataset.py --repo-id franka_fr3_softtoy_in_drawer --root ./datasets/franka_fr3_softtoy_in_drawer
    
    # Visualize pusht dataset from hub
    python visualize_dataset.py --repo-id lerobot/pusht
    
    # Save visualization for later viewing
    python visualize_dataset.py --repo-id my_dataset --episode-index 0 --save --output-dir ./outputs
    # Then view with: rerun ./outputs/my_dataset_episode_0.rrd
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_episode(
    repo_id: str,
    dataset_root: Optional[str],
    episode_index: int,
    save: bool = False,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    mode: str = "local",
) -> None:
    """
    Visualize a single episode of the dataset.
    
    Args:
        repo_id: Repository ID of the dataset
        dataset_root: Path to the dataset root directory (None for hub datasets)
        episode_index: Index of the episode to visualize
        save: Whether to save the visualization to a file
        output_dir: Directory to save the visualization file
        batch_size: Batch size for data loading
        mode: Visualization mode ("local" or "distant")
    """
    logger.info(f"Loading dataset: {repo_id}")
    if dataset_root:
        logger.info(f"Dataset root: {dataset_root}")
    else:
        logger.info("Loading from Hugging Face hub")
    logger.info(f"Visualizing episode {episode_index}")
    
    try:
        # Load the dataset for the specific episode
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_root,
            episodes=[episode_index],
        )
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Total frames in episode {episode_index}: {len(dataset)}")
        logger.info(f"Dataset info:")
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
            logger.info("To view the saved visualization, run:")
            logger.info(f"  rerun {output_path}")
        else:
            logger.info("Visualization opened in Rerun viewer")
            
    except Exception as e:
        logger.error(f"Error visualizing episode {episode_index}: {e}")
        raise


def visualize_multiple_episodes(
    repo_id: str,
    dataset_root: Optional[str],
    episode_indices: List[int],
    save: bool = False,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    delay_between_episodes: float = 2.0,
) -> None:
    """
    Visualize multiple episodes in sequence.
    
    Args:
        repo_id: Repository ID of the dataset
        dataset_root: Path to the dataset root directory (None for hub datasets)
        episode_indices: List of episode indices to visualize
        save: Whether to save the visualizations to files
        output_dir: Directory to save the visualization files
        batch_size: Batch size for data loading
        delay_between_episodes: Delay in seconds between episodes
    """
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
            
            # Add delay between episodes if visualizing multiple
            if i < len(episode_indices) - 1 and not save:
                logger.info(f"Waiting {delay_between_episodes}s before next episode...")
                time.sleep(delay_between_episodes)
                
        except Exception as e:
            logger.error(f"Failed to visualize episode {episode_idx}: {e}")
            continue


def get_dataset_info(repo_id: str, dataset_root: Optional[str]) -> None:
    """
    Print information about the dataset.
    
    Args:
        repo_id: Repository ID of the dataset
        dataset_root: Path to the dataset root directory (None for hub datasets)
    """
    try:
        # Load just the metadata without episodes
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_root,
            episodes=[0],  # Load just one episode to get metadata
        )
        
        print(f"\n=== Dataset Information ===")
        print(f"Repository ID: {repo_id}")
        print(f"Robot type: {dataset.meta.robot_type}")
        print(f"Total episodes: {dataset.meta.total_episodes}")
        print(f"Total frames: {dataset.meta.total_frames}")
        print(f"FPS: {dataset.meta.fps}")
        print(f"Camera keys: {dataset.meta.camera_keys}")
        print(f"Video keys: {dataset.meta.video_keys}")
        
        # Print feature information
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
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Visualize any LeRobot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset (e.g., 'lerobot/pusht' or 'my_dataset')"
    )
    
    parser.add_argument(
        "--episode-index",
        type=int,
        nargs="*",
        default=[0],
        help="Episode index(es) to visualize. Can specify multiple episodes. Default: [0]"
    )
    
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Visualize all episodes in the dataset (use with caution)"
    )
    
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory of the dataset for local datasets. Leave empty for hub datasets."
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save visualization to .rrd file instead of opening viewer"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualization_outputs",
        help="Directory to save visualization files when --save is used. Default: ./visualization_outputs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loading. Default: 32"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "distant"],
        default="local",
        help="Visualization mode. Default: local"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between episodes when visualizing multiple. Default: 2.0"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset information and exit"
    )
    
    args = parser.parse_args()
    
    # Validate dataset path if local
    if args.root:
        dataset_path = Path(args.root)
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            logger.error("Make sure you have the dataset in the correct location.")
            return
    
    # Show dataset info if requested
    if args.info:
        get_dataset_info(args.repo_id, args.root)
        return
    
    # Determine which episodes to visualize
    if args.all_episodes:
        # Get total episodes from dataset metadata
        try:
            temp_dataset = LeRobotDataset(
                repo_id=args.repo_id,
                root=args.root,
                episodes=[0],
            )
            episode_indices = list(range(temp_dataset.meta.total_episodes))
            logger.warning(f"Visualizing ALL {len(episode_indices)} episodes. This may take a while!")
        except Exception as e:
            logger.error(f"Could not determine total episodes: {e}")
            return
    else:
        episode_indices = args.episode_index
    
    # Validate episode indices
    try:
        temp_dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.root,
            episodes=[0],
        )
        max_episode = temp_dataset.meta.total_episodes - 1
        for episode_idx in episode_indices:
            if episode_idx < 0 or episode_idx > max_episode:
                logger.error(f"Episode index {episode_idx} is out of range [0, {max_episode}]")
                return
    except Exception as e:
        logger.error(f"Could not validate episode indices: {e}")
        return
    
    # Create output directory if saving
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizations will be saved to: {output_dir}")
    
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