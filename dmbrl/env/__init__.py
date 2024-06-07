from dmbrl.env.basicCar import basicCarEnv
from dmbrl.env.point import PointEnv
from gym.envs.registration import register

register(
    id='basicCar',
    entry_point='dmbrl.env.basicCar:basicCarEnv'
)

register(
    id='point',
    entry_point='dmbrl.env.point:PointEnv'
)