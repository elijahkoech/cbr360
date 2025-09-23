# notebooks/setup_notebook.py

import sys
from pathlib import Path

def setup_project_path():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.append(str(root))
