import gym
from gym import spaces
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from ue_uav_bs.agents import UE, UAV

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None):

        self.world = world
        # modified 2
        self.agents = []
        self.agents.append(self.world.uav_cluster)
        self.agents.append(self.world.ue_cluster)
        # set required vectorized gym env property
        # modified 2
        self.n = self.world.uav_num + self.world.ue_num
        # self.n = self.world.uav_num
        # self.n = self.world.ue_num
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        # environment parameters
        self.discrete_action_space = True
        # self.discrete_action_space = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        # modified 2
        for agent_index in range(0, 2):
        # for agent_index in range(0, 1):
            for agent in self.agents[agent_index]:
                total_action_space = []
                if isinstance(agent, UAV.UAV):
                    self.discrete_action_space = False
                elif isinstance(agent, UE.UE):
                    self.discrete_action_space = True
                # physical action space
                if self.discrete_action_space:
                    # modified 2
                    if isinstance(agent, UAV.UAV):
                        u_action_space = spaces.Discrete(world.action_dim_for_uav)
                    elif isinstance(agent, UE.UE):
                        division_num_of_one_ue = (1 + len(world.uav_cluster) + len(world.bs_cluster) + len(world.uav_cluster) * len(
                            world.bs_cluster))
                        u_action_space = spaces.Discrete(division_num_of_one_ue)
                    else:
                        u_action_space = 1
                else:
                    u_action_space = spaces.Box(low=-1.0, high=+1.0, shape=(world.action_dim_for_uav,), dtype=np.float32)
                total_action_space.append(u_action_space)
                # total action space
                if len(total_action_space) > 1:
                    # all action spaces are discrete, so simplify to MultiDiscrete action space
                    if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                        act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                    else:
                        act_space = spaces.Tuple(total_action_space)
                    self.action_space.append(act_space)
                else:
                    self.action_space.append(total_action_space[0])
                # observation space
                obs_dim = len(observation_callback(agent, self.world))
                self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

    def step(self, action_n):
        obs_n = []
        # modified 2
        self.agents = []
        self.agents.append(self.world.uav_cluster)
        self.agents.append(self.world.ue_cluster)
        # set action for each agent
        # for i, agent in enumerate(self.agents):
        #     self._set_action(action_n[i], self.action_space[i])
        for i in range(0, len(action_n)):
            self._set_action(action_n[i], self.action_space[i])
        # advance world state
        reward_n, is_terminal_n, delay_n, situation_n = self.world.step(self.agents, action_n)
        # record observation for each agent
        # modified 2
        for agent_index in range(0, 2):
        # for agent_index in range(0, 1):
            for agent in self.agents[agent_index]:
                obs_n.append(self._get_obs(agent))

        return obs_n, reward_n, is_terminal_n, delay_n, situation_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        # modified 2
        self.agents = []
        self.agents.append(self.world.uav_cluster)
        self.agents.append(self.world.ue_cluster)
        # modified 2
        for agent_index in range(0, 2):
        # for agent_index in range(0, 1):
            for agent in self.agents[agent_index]:
                obs_n.append(self._get_obs(agent))
        return obs_n

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, action_n):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(self.world, self.agents, action_n)

    # set env action for a particular agent
    def _set_action(self, action, action_space, time=None):
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]
