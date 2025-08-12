import yaml
import importlib


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_class(model_name: str):
    """
    Dynamically import a model from models/ by name.
    Example: get_model_class("baseline_model") -> models.baseline_model.BaselineModel
    """
    module = importlib.import_module(f"models.{model_name}")
    # Assumes the main class in the model file is the first class defined
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type):
            return obj
    raise ImportError(f"No class found in models/{model_name}.py")
