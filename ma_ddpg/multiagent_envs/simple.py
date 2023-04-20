from ma_ddpg.multiagent_envs.core import World
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.reset()

    def reward(self, world, agents, action_n):
        reward_n, is_terminal_n, delay, situation_n = world.step(agents, action_n)
        return reward_n

    def observation(self, agent, world):
        return world._get_obs(agent)
