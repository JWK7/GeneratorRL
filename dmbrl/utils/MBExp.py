import dmbrl.utils.Agent as Agent
class MBExperiment:
    def __init__(self,params):
        self.params = params

    def run_experiment(self):
        for i in range(self.params.exp_cfg.iteration):
            agent = Agent.Agent(self.params)
            agent.train()
        agent.test()
        pass