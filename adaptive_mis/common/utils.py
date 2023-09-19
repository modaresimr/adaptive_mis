
# Arg Max in a Dic
def arg_max_dict(dic):
    mx = {'v': 0, 'i': 0}
    for d in dic:
        tmp = dic[d]
        if (mx['v'] < tmp):
            mx['v'] = tmp
            mx['i'] = d
    return mx['i']


def class_by_name(clazz):
    module_name, class_name = clazz.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def loader(config, module=None, **other_args):
    cfg = config[module].copy() if module else config.copy()
    params = cfg['params'] or {}
    item = class_by_name(cfg.pop('class'))(**{**params, **other_args})
    item.title = cfg['title']
    return item


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


from termcolor import colored


def printc(string, p=None, end='\n'):
    if not p:
        print(string)
        return
    pre = f"{bcolors.ENDC}"

    if "bold" in p.lower():
        pre += bcolors.BOLD
    elif "underline" in p.lower():
        pre += bcolors.UNDERLINE
    elif "header" in p.lower():
        pre += bcolors.HEADER

    if "warning" in p.lower():
        pre += bcolors.WARNING
    elif "error" in p.lower():
        pre += bcolors.FAIL
    elif "ok" in p.lower():
        pre += bcolors.OKGREEN
    elif "info" in p.lower():
        if "blue" in p.lower():
            pre += bcolors.OKBLUE
        else:
            pre += bcolors.OKCYAN

    print(f"{pre}{string}{bcolors.ENDC}", end=end)


def seed_worker(worker_id):
    import torch
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
