# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import glob
import os.path
import os
import numpy as np

def process_raw_data(force_pre_processing_overwrite=True):
    print("Starting pre-processing of raw data...")
    input_files = glob.glob("data/raw/*")
    input_files_filtered = filter_input_files(input_files)
    for file_path in input_files_filtered:
        filename = file_path.split('/')[-1]
        preprocessed_file_name = "data/preprocessed/"+filename+".npy"

        # check if we should remove the any previously processed files
        if os.path.isfile(preprocessed_file_name):
            print("Preprocessed file for " + filename + " already exists.")
            if force_pre_processing_overwrite:
                print("force_pre_processing_overwrite flag set to True, overwriting old file...")
                os.remove(preprocessed_file_name)
            else:
                print("Skipping pre-processing for this file...")

        if not os.path.isfile(preprocessed_file_name):
            process_file(filename, preprocessed_file_name)
    print("Completed pre-processing.")


def process_file(input_file, output_file):
    print("Processing raw data file", input_file)
    output_data = []
    np.save(output_file, output_data)
    print("Wrote output to", output_file)


def filter_input_files(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))