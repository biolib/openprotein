"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import os
import subprocess
import sys

from op_cli import main
from preprocessing import process_raw_data

def test():

    process_raw_data(False, raw_data_path="tests/data/raw/*",
                     force_pre_processing_overwrite=True)

    # find original and transformed coordinates
    """origcoords = pos.numpy()
    origcoords = np.resize(origcoords, (len(origcoords) * 3, 3))
    write_pdb("origcoords.pdb", protein_id_to_str(prim.tolist()), origcoords)
    transf = tertiary.numpy()
    transf = np.resize(transf, (len(transf) * 3, 3))
    write_pdb("transf.pdb", protein_id_to_str(prim.tolist()), transf)
    sup = SVDSuperimposer()
    sup.set(transf, origcoords)
    sup.run()
    # rotation and transformation for the superimposer
    #rot, tran = sup.get_rotran()
    #print(rot, tran)
    rms = sup.get_rms()
    print("RMS", rms)
    # The segment below finds the structure of the orignal coordinates and the transformed
    encoded = prim.tolist()
    pos_angles = calculate_dihedral_angles(torch.squeeze(pos), use_gpu)
    ter_angles = calculate_dihedral_angles(tertiary, use_gpu)
    pos_struc = get_structure_from_angles(encoded, pos_angles)
    ter_struc = get_structure_from_angles(encoded, ter_angles)
    write_to_pdb(pos_struc, "transformed")
    write_to_pdb(ter_struc, "original")"""


    sys.argv = ["__main__.py", "--min-updates", "1", "--eval-interval", "1",
                "--experiment-id", "rrn", "--hide-ui",
                "--file", "data/preprocessed/testfile.txt.hdf5"]
    main()

    path_to_onnx_file = './tests/output/openprotein.onnx'
    if os.path.exists(path_to_onnx_file):
        os.remove(path_to_onnx_file)
    sub_process = subprocess.Popen(["pipenv", "run", "python", "./tests/onnx_export.py"])
    stdout, stderr = sub_process.communicate()
    print(stdout, stderr)
    assert sub_process.returncode == 0
    assert os.path.exists(path_to_onnx_file)
