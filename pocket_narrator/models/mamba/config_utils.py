import yaml

def load_yaml(path: str):
    if path is None:
        raise ValueError("load_yaml got None – did you forget to pass the config path?")
    with open(path, "r") as f:
        return yaml.safe_load(f)

