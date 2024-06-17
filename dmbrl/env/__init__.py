from dmbrl.env.basicCar import basicCarEnv
from dmbrl.env.point import PointEnv
from dmbrl.env.dampened_humanoid import HumanoidEnv
from dmbrl.env.dampened_pusher import PusherEnv

from gym.envs.registration import register

register(
    id='basicCar',
    entry_point='dmbrl.env.basicCar:basicCarEnv'
)

register(
    id='dampened_pusher',
    entry_point='dmbrl.env.dampened_pusher:PusherEnv'
)

register(
    id="dampened_humanoid",
    entry_point="dmbrl.env.dampened_humanoid:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id='point',
    entry_point='dmbrl.env.point:PointEnv'
)