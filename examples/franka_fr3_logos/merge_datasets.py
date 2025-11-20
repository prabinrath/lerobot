#!/usr/bin/env python
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
