"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""
import argparse
import torch
from preprocessing import process_raw_data
from util import write_out

print("------------------------")
print("--- OpenProtein v0.1 ---")
print("------------------------")


def main():
    parser = argparse.ArgumentParser(description="OpenProtein version 0.1")
    parser.add_argument('--no_force_pre_processing_overwrite',
                        dest='no_force_pre_processing_overwrite',
                        action='store_false',
                        help='Force overwrite existing preprocessed files', default=True)
    args, _unknown = parser.parse_known_args()

    uge_gpu = False
    if torch.cuda.is_available():
        write_out("CUDA is available, using GPU")
        uge_gpu = True

    process_raw_data(uge_gpu, force_pre_processing_overwrite=args.force_pre_processing_overwrite)


main()
