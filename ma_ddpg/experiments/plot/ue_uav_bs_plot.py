""" 3D示意图绘制 """


import matplotlib.pyplot as plt
import numpy as np
import math


# 动态绘图
def plot(uav_cluster_x, uav_cluster_y, uav_cluster_z, ue_cluster, bs_cluster, irs_cluster, time_array, target_slice_array, building_cluster, reward, episode, steps):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = plt.subplot(1, 2, 1, projection='3d')  # 设置3D绘图空间
    plt.title('UAV飞行轨迹与GU-BS相对位置示意图 \n\n ------- Reward: %7.3f' % reward[0] + ", Episode: " + str(episode[0]) + ", Steps: " + str(steps[0]))  # 添加标题及相关信息注解
    plt.xlabel('x')  # 给横轴命名
    plt.ylabel('y')  # 给纵轴命名
    # 在下面几个坐标的位置设置白点，使初始绘图范围包括这些点，一定程度上避免动态图坐标轴缩放
    ax.plot(100, 0, 0, c='white')
    ax.plot(0, 100, 0, c='white')
    ax.plot(0, 0, 100, c='white')
    ax.plot(0, 0, 0, c='white')

    # bx = plt.subplot(1, 2, 2)  # 设置2D绘图空间
    # plt.xlabel('时隙数')  # 给横轴命名
    # plt.ylabel('delay')  # 给纵轴命名
    # plt.title('优化指标delay')  # 添加标题
    # 在下面坐标的位置设置白点，使初始绘图范围包括该点，避免动态图坐标轴缩放
    # 使用plt.ylim指定y轴纵坐标范围
    plt.xlim(0, 600)
    plt.ylim(0, 400)

    # 绘制IRS 示意图
    x_irs = []
    y_irs = []
    z_irs = []

    for irs in irs_cluster:
        x_irs.append(irs.x)
        y_irs.append(irs.y)
        z_irs.append(irs.z)

    uav_start_x = 0
    uav_start_y = 100
    uav_start_z = 0
    uav_destination_x = 600
    uav_destination_y = 400
    uav_destination_z = 0

    ax.scatter3D(x_irs, y_irs, z_irs, 'darkslateblue', marker='^')
    ax.scatter3D(uav_start_x, uav_start_y, uav_start_z, 'deeppink', marker='s')
    ax.scatter3D(uav_destination_x, uav_destination_y, uav_destination_z, 'deeppink', marker='d')

    # 绘制 ue 示意图，用有高度的线段和一个球体进行表示
    for ue in ue_cluster:
        x = [ue.x, ue.x]
        y = [ue.y, ue.y]
        z = [0, ue.z]
        ax.plot(x, y, z, color="red")
        # center and radius
        center = [ue.x, ue.y, ue.z + 2]
        radius = 2
        # data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        # surface plot rstride 值越大，图像越粗糙
        ax.plot_surface(x, y, z, rstride=3, cstride=3, color="red", alpha=0.5)
    # 绘制 bs 示意图，用有高度的线段进行表示 KEY 已经将基站的图形换为三棱锥进行表示
    for bs in bs_cluster:
        x = [bs.x, bs.x, bs.x - math.sqrt(3) * 2, bs.x + math.sqrt(3) * 2]
        y = [bs.y, bs.y + 3, bs.y - 3, bs.y - 3]
        z = [bs.z, 0, 0, 0]
        for i in range(4):
            for j in range(4):
                ax.plot((x[i], x[j]), (y[i], y[j]), (z[i], z[j]), color='cadetblue', alpha=0.5)

    # 绘制建筑物 building 的立方体
    for building in building_cluster:
        kwargs = {'alpha': 0.5, 'color': "slategray"}
        # 需要指定间隔起始点、终止端，以及指定分隔值总数（包括起始点和终止点）；最终函数返回间隔类均匀分布的数值序列。
        xx = np.linspace(building.x, building.x + building.dx, 2)
        yy = np.linspace(building.y, building.y + building.dy, 2)
        zz = np.linspace(building.z, building.z + building.dz, 2)
        xx2, yy2 = np.meshgrid(xx, yy)
        ax.plot_surface(xx2, yy2, np.full_like(xx2, building.z), **kwargs)
        ax.plot_surface(xx2, yy2, np.full_like(xx2, building.z + building.dz), **kwargs)
        yy2, zz2 = np.meshgrid(yy, zz)
        ax.plot_surface(np.full_like(yy2, building.x), yy2, zz2, **kwargs)
        ax.plot_surface(np.full_like(yy2, building.x + building.dx), yy2, zz2, **kwargs)
        xx2, zz2 = np.meshgrid(xx, zz)
        ax.plot_surface(xx2, np.full_like(yy2, building.y), zz2, **kwargs)
        ax.plot_surface(xx2, np.full_like(yy2, building.y + building.dy), zz2, **kwargs)

    # 绘制动态路线图 + 动态 delay 折线图
    x, y, z, xx, yy = [], [], [], [], []
    rows_index = -1
    for rows in uav_cluster_x:
        rows_index += 1
        step_index = -1
        while True:
            step_index += 1
            if step_index >= len(uav_cluster_x[rows_index]):
                break

            # 动态路线图
            # 设置x轴坐标
            x.append(uav_cluster_x[rows_index][step_index])
            # 设置y轴坐标
            y.append(uav_cluster_y[rows_index][step_index])
            # 设置z轴坐标
            z.append(uav_cluster_z[rows_index][step_index])
            ax.plot(x, y, z, c='C0')  # 绘制对应连线的三维线性图

            # 动态 delay 折线图
            # if step_index >= 22 and step_index <= len(time_array[rows_index]) + 21:
            #     # 设置x轴坐标
            #     xx.append(time_array[rows_index][step_index - 22])
            #     # 设置y轴坐标
            #     yy.append(target_slice_array[rows_index][step_index - 22])
            #     bx.plot(xx, yy, c='C0')  # 绘制对应连线的二维线性图
            plt.pause(0.001)

    plt.grid()
    plt.show()


# 静态绘图
# def plot_static(uav_cluster_x, uav_cluster_y, uav_cluster_z, ue_cluster, bs_cluster, time_array, target_slice_array, building_cluster, reward, episode, steps):
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
#     plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
#
#     ax = plt.subplot(1, 2, 1, projection='3d')  # 设置3D绘图空间
#     plt.title('UAV飞行轨迹与UE-BS相对位置示意图 \n\n ------- Reward: %7.3f' % reward[0] + ", Episode: " + str(episode[0]) + ", Steps: " + str(steps[0]))  # 添加标题及相关信息注解
#     plt.xlabel('x')  # 给横轴命名
#     plt.ylabel('y')  # 给纵轴命名
#     # 在下面几个坐标的位置设置白点，使初始绘图范围包括这些点，一定程度上避免动态图坐标轴缩放
#     ax.plot(100, 0, 0, c='white')
#     ax.plot(0, 100, 0, c='white')
#     ax.plot(0, 0, 100, c='white')
#     ax.plot(0, 0, 0, c='white')
#
#     bx = plt.subplot(1, 2, 2)  # 设置2D绘图空间
#     plt.xlabel('时隙数')  # 给横轴命名
#     plt.ylabel('优化指标（delay）在离散时间上的微分值')  # 给纵轴命名
#     plt.title('单一时隙上的优化指标（delay）微分值')  # 添加标题
#     # 在下面坐标的位置设置白点，使初始绘图范围包括该点，避免动态图坐标轴缩放
#     # 使用plt.ylim指定y轴纵坐标范围
#     plt.xlim(0, 600)
#     plt.ylim(0, 20)
#
#     # 绘制 ue 示意图，用有高度的线段和一个球体进行表示
#     for ue in ue_cluster:
#         x = [ue.x, ue.x]
#         y = [ue.y, ue.y]
#         z = [0, ue.z]
#         ax.plot(x, y, z, color="red")
#         # center and radius
#         center = [ue.x, ue.y, ue.z+2]
#         radius = 2
#         # data
#         u = np.linspace(0, 2 * np.pi, 100)
#         v = np.linspace(0, np.pi, 100)
#         x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
#         y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
#         z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
#         # surface plot rstride 值越大，图像越粗糙
#         ax.plot_surface(x, y, z, rstride=3, cstride=3, color="red", alpha=0.5)
#     # 绘制 bs 示意图，用有高度的线段进行表示 KEY 已经将基站的图形换为三棱锥进行表示
#     for bs in bs_cluster:
#         # x = [bs.x, bs.x]
#         # y = [bs.y, bs.y]
#         # z = [0, bs.z]
#         # ax.plot(x, y, z, color="green")
#         x = [bs.x, bs.x, bs.x-math.sqrt(3)*2, bs.x+math.sqrt(3)*2]
#         y = [bs.y, bs.y+3, bs.y-3, bs.y-3]
#         z = [bs.z, 0, 0, 0]
#         for i in range(4):
#             for j in range(4):
#                 ax.plot((x[i], x[j]), (y[i], y[j]), (z[i], z[j]), color="cadetblue", alpha=0.5)
#
#     # 绘制建筑物 building 的立方体
#     for building in building_cluster:
#         kwargs = {'alpha': 0.5, 'color': "slategray"}
#         # xx = [building.x, building.x, building.x+building.dx, building.x+building.dx, building.x]
#         # yy = [building.y, building.y+building.dy, building.y+building.dy, building.y, building.y]
#         # ax.plot3D(xx, yy, [building.z]*5, **kwargs)
#         # ax.plot3D(xx, yy, [building.z+building.dz]*5, **kwargs)
#         # ax.plot3D([building.x, building.x], [building.y, building.y], [building.z, building.z+building.dz], **kwargs)
#         # ax.plot3D([building.x, building.x], [building.y+building.dy, building.y+building.dy], [building.z, building.z+building.dz], **kwargs)
#         # ax.plot3D([building.x+building.dx, building.x+building.dx], [building.y+building.dy, building.y+building.dy], [building.z, building.z+building.dz], **kwargs)
#         # ax.plot3D([building.x+building.dx, building.x+building.dx], [building.y, building.y], [building.z, building.z+building.dz], **kwargs)
#         xx = np.linspace(building.x, building.x + building.dx, 2)
#         yy = np.linspace(building.y, building.y + building.dy, 2)
#         zz = np.linspace(building.z, building.z + building.dz, 2)
#         xx2, yy2 = np.meshgrid(xx, yy)
#         ax.plot_surface(xx2, yy2, np.full_like(xx2, building.z), **kwargs)
#         ax.plot_surface(xx2, yy2, np.full_like(xx2, building.z + building.dz), **kwargs)
#         yy2, zz2 = np.meshgrid(yy, zz)
#         ax.plot_surface(np.full_like(yy2, building.x), yy2, zz2, **kwargs)
#         ax.plot_surface(np.full_like(yy2, building.x + building.dx), yy2, zz2, **kwargs)
#         xx2, zz2 = np.meshgrid(xx, zz)
#         ax.plot_surface(xx2, np.full_like(yy2, building.y), zz2, **kwargs)
#         ax.plot_surface(xx2, np.full_like(yy2, building.y + building.dy), zz2, **kwargs)
#
#     # 绘制静态路线图 + 静态 delay 折线图
#     rows_index = -1
#     for rows in uav_cluster_x:
#         rows_index += 1
#
#         # 静态路线图
#         # 设置x轴坐标
#         x = uav_cluster_x[rows_index]
#         # 设置y轴坐标
#         y = uav_cluster_y[rows_index]
#         # 设置z轴坐标
#         z = uav_cluster_z[rows_index]
#         ax.plot(x, y, z)  # 绘制对应连线的三维线性图
#
#         # 静态 delay 折线图
#         # 设置x轴坐标
#         xx = time_array[rows_index]
#         # 设置y轴坐标
#         yy = target_slice_array[rows_index]
#         bx.plot(xx, yy)  # 绘制对应连线的二维线性图
#
#     plt.grid()
#     plt.show()
