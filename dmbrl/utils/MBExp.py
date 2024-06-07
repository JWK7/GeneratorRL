import dmbrl.utils.Agent as Agent
import time
from dmbrl.utils.DataFunctions import printProgressBar
    
class MBExperiment:
    def __init__(self,params):
        self.params = params

    def run_experiment(self):

        printProgressBar(0, self.params.exp_cfg.iteration, prefix = 'Progress:', suffix = 'Complete', length = 50)
        agent = Agent.Agent(self.params)
        for i in range(self.params.exp_cfg.iteration):
            agent.train()
            printProgressBar(i + 1, self.params.exp_cfg.iteration, prefix = 'Progress:', suffix = 'Complete', length = 50)
        agent.test()
        pass