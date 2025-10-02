import os, sys, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.join(ROOT, "model", "core.py")

if not os.path.exists(CORE_PATH):
    raise FileNotFoundError(f"Could not find core.py at {CORE_PATH}")

spec = importlib.util.spec_from_file_location("sharpmodel.core", CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

if __name__ == "__main__":
    core.run("config.yaml")
