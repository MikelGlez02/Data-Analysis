import sys

def check_python_version():
    if sys.version_info < (3, 7):
        raise RuntimeError("Python 3.7 or higher is required for this project.")
