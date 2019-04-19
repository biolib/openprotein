from preprocessing import * 
import torch
import argparse

print("------------------------")
print("--- OpenProtein v0.1 ---")
print("------------------------")

parser = argparse.ArgumentParser(description = "OpenProtein version 0.1")
parser.add_argument('--no_force_pre_processing_overwrite', dest='no_force_pre_processing_overwrite', action='store_false',
                    help='Force overwrite existing preprocessed files', default=True)
args, unknown = parser.parse_known_args()

use_gpu = False
if torch.cuda.is_available():
    write_out("CUDA is available, using GPU")
    use_gpu = True

process_raw_data(use_gpu, force_pre_processing_overwrite=args.force_pre_processing_overwrite)