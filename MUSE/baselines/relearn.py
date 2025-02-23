import sys
import pathlib
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(BASELINE_PATH)

from baselines import relearn

import argparse
from os.path import basename, dirname, join as pathjoin


def main():
    args = get_args()
    
    relearn_model_dir = pathjoin(dirname(args.model_dir), basename(args.model_dir) + "_relearn_" + str(args.max_steps))
    relearn(
        args.model_dir, args.data_file, relearn_model_dir,
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.lr,
        max_len=args.max_len,
        tokenizer_dir=args.tokenizer_dir
    )

    return


def get_args():
    parser = argparse.ArgumentParser(description="Unlearning baselines")
    
    parser.add_argument(
        '--model_dir', type=str,
        help="Path to the target model's hf directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default=None,
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory."
    )
    parser.add_argument(
        '--data_file', type=str,
        help="Path to the forget set file."
    )

    parser.add_argument(
        '--max_len', type=int, default=4096,
        help="max length of input ids fed to the model"
    )
    parser.add_argument(
        '--resume_from_checkpoint', action='store_true',
    )

    # Gradient ascent & Gradient difference
    parser.add_argument('--per_device_batch_size', type=int, default=2)

    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help="Learning rate if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )
    parser.add_argument(
        '--max_steps', type=int, default=25,
        help="Number of max_steps of training if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
