import sys
import os
sys.path.append(os.getcwd())

import argparse
from dotmap import DotMap
from dmbrl.config import *

from dmbrl.utils.MBExp import MBExperiment

def main(env,ctrl_args):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})

    # cfg = create_config(env, ctrl_args)
    cfg = create_config(env,ctrl_args)

    exp = MBExperiment(cfg)
    exp.run_experiment()
    pass

if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-env',type=str, required = True)
    parse.add_argument('-ca', '--ctrl_arg', action= 'append', nargs=2, default = [])
    args=parse.parse_args()

    main(args.env,args.ctrl_arg)
