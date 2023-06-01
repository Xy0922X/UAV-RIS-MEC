import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import xlwt

import ma_ddpg.maddpg.common.tf_util as U
from ma_ddpg.maddpg.trainer.maddpg_renamed import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from ma_ddpg.multiagent_envs.state_normalizetion import StateNormalization

s_normal = StateNormalization()


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=3000, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=8, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="exp", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # 该模型将观测值作为输入，并返回所有动作的值
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env():
    from ma_ddpg.multiagent_envs.environment import MultiAgentEnv
    from ma_ddpg.multiagent_envs import simple
    # load scenario from script
    scenario = simple.Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    # if benchmark:
    #     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    # else:
    #     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # return env
    return MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    # 训练
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env()
        # 返回(world, scenario.reset_world, scenario.reward, scenario.observation)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        #  将每个智能体的信息存储在列表 obs_shape_n
        num_adversaries = min(env.n, arglist.num_adversaries)
        # 记录对抗智能体的个数，这里没用上
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        #  训练 返回数组trainers[]
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # 所有智能体的奖励总和
        agent_rewards = [[0.0] for _ in range(env.n)]  # 第n个智能体的奖励总和
        final_ep_rewards = []  # 训练曲线的奖励总和
        final_ep_ag_rewards = []  # 智能体奖励训练曲线
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        t2 = time.time()
        format_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        episodes_data_workbook = xlwt.Workbook()
        reward_data_workbook = xlwt.Workbook()
        sheet = reward_data_workbook.add_sheet('ep_reward_list')
        sheet.write(0, 0, 'Episode')
        sheet.write(0, 1, 'Reward')
        sheet.write(0, 2, 'situation')
        ep_reward = 0
        ep_delay = 0
        target_slice_array = []
        action_array = [[] for m in range(len(env.world.uav_cluster))]
        energy_array = [[] for n in range(len(env.world.uav_cluster))]
        ep_reward_list = []
        situation_n_list = []
        train_step_temp = 0
        ep_reward_temp = 0

        print('Starting iterations...')
        while True:
            # 获取多个智能体在当前状态下的动作
            action_n = [agent.action(s_normal.state_normal(obs)) for agent, obs in zip(trainers, obs_n)]
            # 状态更新
            new_obs_n, rew_n, is_terminal_n, delay_n, situation_n = env.step(action_n)
            for i in range(len(rew_n)):
                rew_n[i] = rew_n[i] / 10000
            episode_step += 1
            # 检查是否所有的智能体都处于终止状态
            done = all(is_terminal_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # 经验回放过程，这个经验池会被用来随机采样，生成一批用于训练神经网络的数据。
            for i, agent in enumerate(trainers):
                agent.experience(s_normal.state_normal(obs_n[i]), action_n[i], rew_n[i],
                                 s_normal.state_normal(new_obs_n[i]), done, terminal)
            obs_n = new_obs_n

            # 记录智能体在当前回合（episode）中的奖励
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew  # 用-1表示当前正在进行的回合
                agent_rewards[i][-1] += rew

            # ep_bs_task_sum += sum(bs_task_sum)
            ep_reward += sum(rew_n)
            ep_delay += sum(delay_n)
            target_slice_array.append(delay_n)
            uav_index = -1
            for uav in env.world.uav_cluster:
                uav_index += 1
                action_array[uav_index].append(action_n[uav_index])
                energy_array[uav_index].append(env.world.uav_cluster[uav_index].energy)

            # 用于保存模型和输出训练过程中的统计信息。
            # if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            if done and len(episode_rewards) % arglist.save_rate == 0:
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            if terminal or done:
                train_step_for_the_episode = train_step - train_step_temp
                train_step_temp = train_step
                time_array = range(1, train_step_for_the_episode + 1)
                ep_reward_for_the_episode = ep_reward - ep_reward_temp
                ep_reward_temp = ep_reward
                episode_numbers = len(episode_rewards)
                # print('Episode:', episode_numbers, ' Steps: %2d' % (train_step_for_the_episode + 1), ' Reward: %7.3f' % ep_reward_for_the_episode, flush=True)
                time_cost = time.time() - t2
                ep_reward_list = np.append(ep_reward_list, ep_reward_for_the_episode)
                situation_n_list.append(str(situation_n))
                # 绘制无人机飞行轨迹 & 保存相关数据
                if episode_numbers % arglist.save_rate == 0:
                    uav_index = -1
                    for uav in env.world.uav_cluster:
                        uav_index += 1
                        uav_cluster_x = np.array(env.world.uav_cluster_x, dtype=object)
                        uav_cluster_y = np.array(env.world.uav_cluster_y, dtype=object)
                        uav_cluster_z = np.array(env.world.uav_cluster_z, dtype=object)
                        # 创建存储数据用的 excel sheet，将数据保存至 excel 表格
                        sheet = episodes_data_workbook.add_sheet(
                            'Episode ' + str(episode_numbers) + ' of uav ' + str(uav_index))
                        sheet.write(0, 0, 'uav_cluster_x')
                        for index in np.arange(len(uav_cluster_x[uav_index])):
                            sheet.write(int(index + 1), 0, uav_cluster_x[uav_index][index])
                        sheet.write(0, 1, 'uav_cluster_y')
                        for index in np.arange(len(uav_cluster_y[uav_index])):
                            sheet.write(int(index + 1), 1, uav_cluster_y[uav_index][index])
                        sheet.write(0, 2, 'uav_cluster_z')
                        for index in np.arange(len(uav_cluster_z[uav_index])):
                            sheet.write(int(index + 1), 2, uav_cluster_z[uav_index][index])
                        sheet.write(0, 3, 'time_array')
                        for index in np.arange(1, 22):
                            sheet.write(index, 3, 0)
                        for index in np.arange(len(time_array)):
                            sheet.write(int(index + 22), 3, time_array[index])
                        sheet.write(0, 4, 'target_slice_array')
                        for index in np.arange(1, 22):
                            sheet.write(index, 4, 0)
                        for index in np.arange(len(target_slice_array)):
                            sheet.write(int(index + 22), 4, str(target_slice_array[index]))
                        sheet.write(0, 5, 'ep_reward')
                        sheet.write(1, 5, ep_reward_for_the_episode)
                        sheet.write(0, 6, 'episode')
                        sheet.write(1, 6, episode_numbers)
                        sheet.write(0, 7, 'steps')
                        sheet.write(1, 7, train_step_for_the_episode + 1)
                        sheet.write(0, 8, 'time_cost(s/episode)')
                        sheet.write(1, 8, time_cost)
                        sheet.write(0, 9, 'action_during_the_step')
                        for index in np.arange(1, 22):
                            sheet.write(index, 9, 0)
                        for index in np.arange(len(action_array[uav_index])):
                            sheet.write(int(index + 22), 9, str(action_array[uav_index][index].tolist()))
                        sheet.write(0, 10, 'energy_of_uav' + str(uav_index) + '_after_the_step')
                        for index in np.arange(1, 22):
                            sheet.write(index, 10, 0)
                        for index in np.arange(len(energy_array[uav_index])):
                            sheet.write(int(index + 22), 10, str(energy_array[uav_index][index]))
                    sheet = reward_data_workbook.get_sheet('ep_reward_list')
                    for index in np.arange(episode_numbers - arglist.save_rate, len(ep_reward_list)):
                        sheet.write(int(index + 1), 0, int(index + 1))
                    for index in np.arange(episode_numbers - arglist.save_rate, len(ep_reward_list)):
                        sheet.write(int(index + 1), 1, ep_reward_list[index])
                    for index in np.arange(episode_numbers - arglist.save_rate, len(ep_reward_list)):
                        sheet.write(int(index + 1), 2, situation_n_list[index])
                    # reward_data_workbook.save('plot/reward_data' + '-' + format_time + '.xls')
                if episode_numbers % (arglist.save_rate * 10) == 0:
                    episodes_data_workbook.save('plot/episodes_data' + '-' + format_time + '.xls')
                    reward_data_workbook.save('plot/reward_data' + '-' + format_time + '.xls')
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                t2 = time.time()
                ep_reward = 0
                ep_reward_temp = 0
                ep_bs_task_sum = 0
                ep_delay = 0
                target_slice_array = []
                action_array = [[] for m in range(len(env.world.uav_cluster))]
                energy_array = [[] for n in range(len(env.world.uav_cluster))]

            # increment global step counter
            train_step += 1

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
