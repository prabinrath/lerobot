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

import argparse
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

def main():
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets")
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        required=True,
        help="Names of datasets to merge (space-separated)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="multi_task_1",
        help="Name for the merged dataset (default: multi_task_1)"
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="./datasets",
        help="Root directory containing datasets (default: ./datasets)"
    )
    
    args = parser.parse_args()
    datasets_dir = Path(args.datasets_dir).resolve()
    
    print("Loading datasets...")
    datasets = []
    for folder in args.dataset_names:
        dataset_path = datasets_dir / folder
        if dataset_path.exists():
            print(f"  Loading {folder}")
            ds = LeRobotDataset(folder, root=dataset_path)
            datasets.append(ds)
            print(f"    Episodes: {ds.meta.total_episodes}, Frames: {ds.meta.total_frames}")
        else:
            print(f"  Warning: {folder} not found, skipping")
    
    if not datasets:
        print("Error: No valid datasets found to merge")
        return
    
    output_dir = datasets_dir / args.output_name
    print(f"\nMerging {len(datasets)} datasets into {args.output_name}...")
    
    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=args.output_name,
        output_dir=output_dir,
    )
    
    print(f"\nMerge complete!")
    print(f"  Total episodes: {merged.meta.total_episodes}")
    print(f"  Total frames: {merged.meta.total_frames}")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    main()
