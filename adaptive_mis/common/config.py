import yaml
import os
from . import printc


def include_constructor(loader, node):
    filepath = loader.construct_scalar(node)
    filepath = os.path.join(*filepath.split('\\'))
    with open(filepath, 'r') as input_file:
        return yaml.load(input_file, Loader=yaml.FullLoader)


# Register the custom constructor
yaml.add_constructor('!include', include_constructor, Loader=yaml.FullLoader)


# class Loader(yaml.SafeLoader):

#     def __init__(self, stream):
#         self._root = os.path.split(stream.name)[0]
#         super().__init__(stream)

#     def include(self, node):
#         filename = os.path.join(self._root, self.construct_scalar(node))
#         with open(filename, 'r') as f:
#             return yaml.load(f, Loader)


# Loader.add_constructor('!include', Loader.include)


def load_config(config_filepath):
    try:
        if os.path.exists(config_filepath):
            with open(config_filepath, 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
        else:
            data = yaml.load(config_filepath, Loader=yaml.FullLoader)

        return data
    except Exception as e:
        printc(f"Config file not found! <{config_filepath}>", "error_bold")
        printc(e, "error_bold")
        exit(1)


def save_config(config, config_filepath):
    try:
        with open(config_filepath, 'w') as file:
            yaml.dump(config, file)
    except FileNotFoundError:
        printc(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)
