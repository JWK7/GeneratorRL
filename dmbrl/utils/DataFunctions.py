from multiprocessing import Process
import gym
import dmbrl.env
import numpy as np
import torch
from pandas import DataFrame
from multiprocessing import Manager

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def clean_data(raw_data):
    S_0 = []
    S_t = []
    deltaS = []
    for i in raw_data:
        s_0,s_t = (raw_data[i])
        S_0.append(s_0)
        S_t.append(s_t)
        deltaS.append(s_t-s_0)
    return torch.tensor(np.array([S_0,S_t,deltaS]),dtype=torch.float)

def gym_sample(exp_cfg):
    raw_data = thread_sampling(exp_cfg)
    sim_data = clean_data(raw_data)
    return sim_data

def thread_sampling(exp_cfg):
    threads = []
    manager = Manager()
    raw_data = manager.dict()
    for i in range(exp_cfg.batch_size):
        t = Process(target=threadSample,args=(i,exp_cfg,raw_data))
        hi = t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    return raw_data

def threadSample(i,exp_cfg,raw_data):
    env = gym.make(exp_cfg.env_name, render_mode="rgb_array")
    obs,_ = env.reset()
    s_0 = obs
    action = np.random.randint(0,2,exp_cfg.num_generators)
    action[action==0] = -1
    for _ in range(exp_cfg.step): obs,*_ = env.step(action)
    s_t = obs
    raw_data[i] = (s_0,s_t)
    return 