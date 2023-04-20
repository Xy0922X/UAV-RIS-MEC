import numpy as np


# 定义函数，计算给定相位下的信道增益
def channel_gain(phase, distance_1, distance_2):
    wavelength = 3e8 / 5.8e9  # 无线电波长
    antenna_spacing = wavelength / 2  # 天线间距
    return np.abs(1 + np.exp(-1j * 2 * np.pi * distance_1 / wavelength * np.cos(phase))) ** 2 / 2 * np.cos(
        np.pi * antenna_spacing / distance_1) \
           * np.abs(1 + np.exp(-1j * 2 * np.pi * distance_2 / wavelength * np.cos(phase))) ** 2 / 2 * np.cos(
        np.pi * antenna_spacing / distance_2)


# 初始化参数
num_phases = 360  # 相位数
distance_1 = 1000  # 无人机与IRS的距离
distance_2 = 500  # 用户与IRS的距离
num_iterations = 1000  # 迭代次数
step_size = 0.01  # 步长
current_phase = np.random.uniform(0, 2 * np.pi)  # 当前相位
best_phase = current_phase  # 最优相位
best_gain = 0  # 最优增益

# AO算法迭代
for i in range(num_iterations):
    # 计算当前相位下的信道增益
    current_gain = channel_gain(current_phase, distance_1, distance_2)

    # 更新最优相位和增益
    if current_gain > best_gain:
        best_phase = current_phase
        best_gain = current_gain

    # 更新当前相位
    gradient = np.imag(channel_gain(current_phase + np.pi / 2, distance_1, distance_2) * np.conj(
        channel_gain(current_phase, distance_1, distance_2)))
    current_phase = current_phase + step_size * gradient

    # 将相位限制在0到2π之间
    if current_phase > 2 * np.pi:
        current_phase = current_phase - 2 * np.pi
    elif current_phase < 0:
        current_phase = current_phase + 2 * np.pi

# 输出结果
print("最优相位为：{:.2f}度".format(best_phase * 180 / np.pi))
print("最优增益为：{:.2f}".format(best_gain))
