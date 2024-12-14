import sys

def check_python_version():
    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9 or higher is required for this project.")
