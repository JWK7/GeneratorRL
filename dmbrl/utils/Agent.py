import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dmbrl.utils.GeneratorFunctions import UpdateG,UpdateX

import dmbrl.utils.DataFunctions as DataFunctions

class Agent:
    def __init__(self,params):
        self.params = params

    def train(self):
        self.Sample()

    def Sample(self):
        data = self.params.tool_cfg.sample()
        UpdateG(self.params,data)
        # self.params.tool_cfg.UpdateGradienceG(data)

        # self.params.log_cfg.Gloss.append(self.params.tool_cfg.UpdateGradienceG(data[:,:,0]))
        # UpdateX(self.params,data)
        # self.params.tool_cfg.UpdateG()
        # for i in range(self.params.exp_cfg.num_generators):

            # if type(loss) is tuple:
            #     self.params.log_cfg.loss += loss[0].cpu(),
            #     self.params.log_cfg.loss_jacobian += loss[1].cpu(),
            # else:
            #     self.params.log_cfg.loss += loss.cpu(),

    def test(self):
        plt.plot(np.array(self.params.log_cfg.Gloss[5:]))
        plt.savefig('outputs/Gloss.png')
        plt.close()
        plt.plot(np.array(self.params.log_cfg.xloss))
        plt.savefig('outputs/xloss.png')
        plt.close()
        # I0, Ix, deltaI, x = self.params.tool_cfg.sample()

        # generator_count = 1
        # for i in self.params.tool_cfg.nn:
        #     pred = i.GNet(deltaI.squeeze().to(i.device)).detach()[0].cpu()
        #     heatmap = sns.heatmap(pred.numpy())
        #     plt.savefig('outputs/heat_G_'+str(generator_count)+'.png')
        #     plt.close()
        #     np.savetxt('outputs/G_'+str(generator_count)+'.csv', pred.numpy(), delimiter=',')
        #     generator_count += 1
        #     i.saveModels()
        # plt.plot(np.array(self.params.log_cfg.loss))
        # plt.savefig('outputs/loss.png')
        # plt.close()
        


        # if self.params.log_cfg.loss_jacobian:
        #     plt.plot(np.array(self.params.log_cfg.loss_jacobian))
        #     plt.savefig('outputs/loss_jacobian.png')
        #     plt.close()
