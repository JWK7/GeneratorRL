from dotmap import DotMap
import os
import importlib.machinery
import importlib.util

import dmbrl.modeling.Model



# cfg:
#     exp_cfg: (experiment configuration)
#         env
#         iteration
#         alpha
#         beta
#         data_size
#         num_generators
#         ground_truth_generator (optional)
#     tool_cfg: (tool configuration)
#         nn
#
#     log_cfg: (logging configuration)
#         loss
def create_config(env_name,ctrl_arg):
    cfg = DotMap(
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    loader = importlib.machinery.SourceFileLoader(env_name, os.path.join(dir_path, "%s.py" % env_name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    cfg_source = importlib.util.module_from_spec(spec)
    loader.exec_module(cfg_source)
    cfg_module = cfg_source.CONFIG_MODULE()

    _create_cfg(cfg,cfg_module)
    return cfg


def _create_cfg(cfg,cfg_module):
    cfg.exp_cfg = cfg_module.exp_cfg
    cfg.tool_cfg = cfg_module.tool_cfg
    cfg.log_cfg = cfg_module.log_cfg

