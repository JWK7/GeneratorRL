from dmbrl.env.basicCar import basicCarEnv

from gym.envs.registration import register

register(
    id='basicCar',
    entry_point='dmbrl.env.basicCar:basicCarEnv'
)