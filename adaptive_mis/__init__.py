import importlib
import sys
from .common import loader


def reload():
    dic = {module_name: module for module_name, module in sys.modules.items()}
    for module_name, module in dic.items():
        if module_name.startswith('adaptive_mis'):
            # print(module)
            importlib.reload(module)
