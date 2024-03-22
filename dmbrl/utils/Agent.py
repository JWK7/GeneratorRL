import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Agent:
    def __init__(self,params):
        self.params = params

    def train(self):
        self.Sample()

    def Sample(self):
        I0, Ix, deltaI, x = self.params.tool_cfg.sample()
        for i in range(len(self.params.tool_cfg.nn)):
            loss = self.params.tool_cfg.UpdateG(i,self.params.tool_cfg.nn,I0,deltaI,x)
            self.params.log_cfg.loss += loss,

    def test(self):
        I0, Ix, deltaI, x = self.params.tool_cfg.sample()

        generator_count = 1
        for i in self.params.tool_cfg.nn:
            pred = i.GNet(I0.squeeze()).detach()[0]
            heatmap = sns.heatmap(pred.numpy())
            plt.savefig('outputs/heat_G_'+str(generator_count)+'.png')
            plt.close()
            np.savetxt('outputs/G_'+str(generator_count)+'.csv', pred.numpy(), delimiter=',')
            generator_count += 1
        plt.plot(np.array(self.params.log_cfg.loss))
        plt.savefig('outputs/loss.png')

        combined_generator = self.params.tool_cfg.CombineGenerators(0,self.params.tool_cfg.nn,I0,x)[0].detach()
        heatmap= sns.heatmap(combined_generator)
        plt.savefig('outputs/heat_xG_combined.png')
        plt.close()
        np.savetxt('outputs/xG_combined.csv', combined_generator.numpy(), delimiter=',')
        pass
