import os
import sys

import file_util
from constants import PROJECT_ROOT, _curr_time, STATE_DIR
from file_util import list_all_files_in_folder
from tee import Tee

using_debugger = True  # getattr(sys, 'gettrace', None)() is not None


def save_all_py_files(folder_path, save_output=True):
    if using_debugger:
        return _curr_time
    runfile_name = os.path.basename(sys.argv[0])
    python_files_path = os.path.join(STATE_DIR, folder_path, runfile_name, _curr_time, "project_files")
    os.makedirs(python_files_path, exist_ok=True)

    if save_output:
        t = Tee(f"{python_files_path}/output.txt")
        t.__enter__()

    for ext in ["py", "yaml"]:  # copy all files with relevant extension
        python_files_in_dir = list_all_files_in_folder(PROJECT_ROOT, ext, recursively=True)
        file_util.copy_files_while_keeping_structure(python_files_in_dir, PROJECT_ROOT, python_files_path)
    with open(f"{python_files_path}/runscript.txt", 'w') as f:
        f.write(' '.join([runfile_name] + sys.argv[1:]))
    print(f"files saved in: \n {python_files_path}")
    return _curr_time
