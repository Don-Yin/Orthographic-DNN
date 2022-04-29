import os
from pathlib import Path


def routine_settings():
    os.environ["TORCH_HOME"] = str(Path("assets", "model_cache"))
    dir_target = os.path.dirname(os.path.realpath(__file__))
    dir_target = os.path.dirname(dir_target)
    os.chdir(dir_target)
