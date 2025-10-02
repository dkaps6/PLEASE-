import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model.core import run

from model.core import run

if __name__ == "__main__":
    run("config.yaml")
