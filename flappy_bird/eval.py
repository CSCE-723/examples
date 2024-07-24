"""
This is a Command Line Interface (CLI) to call custom_flappy_bird.eval.evaluate with passed args.

For more info, in terminal from the workspace dir, type:
    python eval.py -h
"""

import os
import sys
import pathlib
import argparse

# Add package to PYTHONPATH
from custom_flappy_bird import ROOT_DIR
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
# print('sys.path', sys.path)

def _get_args():
    """Parses the command line arguments and returns them."""

    parser = argparse.ArgumentParser(description=__doc__)
    
    # Argument for the mode of execution (human or random):
    parser.add_argument(
        "checkpoint", 
        type=str, 
        # default="", 
        help="Checkpoint to load Algorithm from for evaluation"
    )
    
    parser.add_argument(
        "--replay_dir",
        "-r",
        type=os.PathLike,
        default=pathlib.Path(__file__).parent.joinpath('replays'),
        help="The folder to save replays to"
    )

    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=20,
        help="Number of episodes to run evaluation"
    )

    parser.add_argument(
        "--workers", 
        "-w", 
        type=int, 
        default=5, 
        help="Number of evaluation workers. Must be <= NUM_CPUs"
        )
    
    return parser.parse_args()

def main():
    args = _get_args()
    print('Replay folder: ', args.replay_dir)
    from custom_flappy_bird.eval import evaluate
    evaluate(
        checkpoint=args.checkpoint,
        replay_folder=args.replay_dir,
        evaluation_duration=args.episodes,
        evaluation_num_workers=args.workers
    )

if __name__ == "__main__":
    main()