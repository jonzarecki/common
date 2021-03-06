import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PROJECT_NAME = os.path.basename(PROJECT_ROOT)
USER_ROOT_DIR = os.path.dirname(os.path.dirname(PROJECT_ROOT))

# consistent for all projects
LONG_TERM_DIR = "/media/yonatanz/yz/"
if not os.path.exists(LONG_TERM_DIR):
    LONG_TERM_DIR = "/cortex/users/jonzarecki/long_term/"

PROJ_LONG_TERM_DIR = os.path.join(LONG_TERM_DIR, PROJECT_NAME)
DATA_LONG_TERM_DIR = os.path.join(LONG_TERM_DIR, "data")
MODEL_LONG_TERM_DIR = os.path.join(LONG_TERM_DIR, "models")

_curr_time = datetime.now().isoformat(' ', 'seconds')

TENSORBOARD_DIR = os.path.join(PROJ_LONG_TERM_DIR, "logs")
STATE_DIR = os.path.join(PROJ_LONG_TERM_DIR, "expr_state")
PROJ_MODELS_LONG_TERM_DIR = os.path.join(PROJ_LONG_TERM_DIR, "models")

# CURRENT_EXPR_DIR = os.path.join(PROJ_LONG_TERM_DIR, os.path.basename(sys.argv[0]), _curr_time)
# SAVED_MODEL_DIR = os.path.join(CURRENT_EXPR_DIR, 'models')
# TMP_EXPR_FILES_DIR = os.path.join(CURRENT_EXPR_DIR, "project_files")
