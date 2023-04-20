import xlrd2
from ue_uav_bs.agents import UE, BS, UAV, Building
from ue_uav_bs.MADDPG.experiments.plot import ue_uav_bs_plot

ue1 = UE.UE(8, 55, 2)
ue2 = UE.UE(50, 25, 2)
ue3 = UE.UE(15, 65, 2)
ue4 = UE.UE(20, 45, 2)
ue5 = UE.UE(65, 85, 2)
uav1 = UAV.UAV(1, 1, 1, 1500000)  # 根据参考论文，无人机初始能量为 500kJ，此处定为 1500kJ 是为了让仿真时无人机可以飞得久一点，到 90 多秒再降落
uav2 = UAV.UAV(90, 100, 1, 1500000)
uav3 = UAV.UAV(90, 10, 1, 1500000)
bs1 = BS.BS(10, 75, 50)
bs2 = BS.BS(110, 85, 50)
bs3 = BS.BS(110, 5, 50)
bs4 = BS.BS(35, 5, 50)
bs5 = BS.BS(60, 70, 50)
# building1 = Building.Building(10, 20, 0, 10, 10, 40)
# building2 = Building.Building(20, 20, 0, 10, 10, 20)
building1 = Building.Building(10, 20, 0, 10, 10, 65)
building2 = Building.Building(20, 20, 0, 10, 10, 50)
building3 = Building.Building(55, 40, 0, 10, 10, 60)
building4 = Building.Building(65, 40, 0, 10, 10, 30)
building5 = Building.Building(55, 30, 0, 10, 10, 20)
building6 = Building.Building(80, 70, 0, 10, 10, 30)
building7 = Building.Building(80, 80, 0, 10, 10, 50)

ue_cluster = [ue2]
# ue_cluster = [ue1, ue2]
# ue_cluster = [ue1, ue2, ue3, ue4, ue5]
# uav_cluster = [uav1]
# uav_cluster = [uav2]
uav_cluster = [uav3]
# uav_cluster = [uav1, uav2]
# bs_cluster = [bs1]
bs_cluster = [bs1, bs2]
# bs_cluster = [bs1, bs2, bs3, bs4, bs5]
building_cluster = [building1, building2, building3, building4, building5, building6, building7]
# building_cluster = [building1, building2]
# building_cluster = [building7]
# building_cluster = []

# # 静态绘图
book = xlrd2.open_workbook("episodes_data-20230319183101.xls")
sheets = []
# episodes_count 用来描述想要绘制的图像所对应的数据页范围
for episodes_count in range(114, 115):
    # offset 用来描述所绘制图像所对应的数据页，实际上此处对应数据页为第 0 + offset*1 或 0 + offset*2、1 + offset*2 页（每次绘制一架无人机的轨迹图像）
    offset = episodes_count
    sheets = []
    # 单无人机数据
    for i in range(0 + offset * 1, 1 + offset * 1):
    # # 双无人机数据
    # for i in range(0 + offset * 2, 2 + offset * 2):
        sheets.append(book.sheets()[i])
    uav_cluster_x = [[] for i in range(6)]
    uav_cluster_y = [[] for i in range(6)]
    uav_cluster_z = [[] for i in range(6)]
    time_array = [[] for i in range(6)]
    target_slice_array = [[] for i in range(6)]
    ep_reward = []
    episode = []
    steps = []
    sheet_index = -1
    for sheet in sheets:
        sheet_index += 1
        i = 0
        for j in range(1, sheets[sheet_index].nrows):
            if sheets[sheet_index].row_values(j)[i] != '':
                uav_cluster_x[sheet_index].append(sheets[sheet_index].row_values(j)[i])

        i = 1
        for j in range(1, sheets[sheet_index].nrows):
            if sheets[sheet_index].row_values(j)[i] != '':
                uav_cluster_y[sheet_index].append(sheets[sheet_index].row_values(j)[i])

        i = 2
        for j in range(1, sheets[sheet_index].nrows):
            if sheets[sheet_index].row_values(j)[i] != '':
                uav_cluster_z[sheet_index].append(sheets[sheet_index].row_values(j)[i])

        i = 3
        for j in range(32, sheets[sheet_index].nrows):
            if sheets[sheet_index].row_values(j)[i] != '':
                time_array[sheet_index].append(sheets[sheet_index].row_values(j)[i])

        i = 4
        if sheet_index == 0 or sheet_index == 1:
            for j in range(32, sheets[sheet_index].nrows):
                if sheets[sheet_index].row_values(j)[i] != '':
                    target_slice_array[sheet_index].append(float(sheets[sheet_index].row_values(j)[i].strip("[]")) - 0.5)
                    # target_slice_array[sheet_index].append(0)
        else:
            for j in range(32, sheets[sheet_index].nrows):
                if sheets[sheet_index].row_values(j)[i] != '':
                    target_slice_array[sheet_index].append(float(sheets[sheet_index].row_values(j)[i].strip("[]")))

        ep_reward.append(sheets[sheet_index].row_values(1)[5])
        episode.append(sheets[sheet_index].row_values(1)[6])
        steps.append(sheets[sheet_index].row_values(1)[7])

    ue_uav_bs_plot.plot_static(uav_cluster_x, uav_cluster_y, uav_cluster_z, ue_cluster, bs_cluster,
                               time_array, target_slice_array, building_cluster, ep_reward, episode, steps)

# # 动态绘图
# book = xlrd2.open_workbook("episodes_data-20230319183101.xls")
# sheets = []
# # episodes_count 用来描述想要绘制的图像所对应的数据页范围
# for episodes_count in range(114, 115):
#     # offset 用来描述所绘制图像所对应的数据页，实际上此处对应数据页为第 0 + offset*1 或 0 + offset*2、1 + offset*2 页（每次绘制一架无人机的轨迹图像）
#     offset = episodes_count
#     sheets = []
#     # 单无人机数据
#     for i in range(0 + offset * 1, 1 + offset * 1):
#     # # 双无人机数据
#     # for i in range(0 + offset * 2, 2 + offset * 2):
#         sheets.append(book.sheets()[i])
#     uav_cluster_x = [[] for i in range(6)]
#     uav_cluster_y = [[] for i in range(6)]
#     uav_cluster_z = [[] for i in range(6)]
#     time_array = [[] for i in range(6)]
#     target_slice_array = [[] for i in range(6)]
#     ep_reward = []
#     episode = []
#     steps = []
#     sheet_index = -1
#     for sheet in sheets:
#         sheet_index += 1
#         i = 0
#         for j in range(1, sheets[sheet_index].nrows):
#             if sheets[sheet_index].row_values(j)[i] != '':
#                 uav_cluster_x[sheet_index].append(sheets[sheet_index].row_values(j)[i])
#
#         i = 1
#         for j in range(1, sheets[sheet_index].nrows):
#             if sheets[sheet_index].row_values(j)[i] != '':
#                 uav_cluster_y[sheet_index].append(sheets[sheet_index].row_values(j)[i])
#
#         i = 2
#         for j in range(1, sheets[sheet_index].nrows):
#             if sheets[sheet_index].row_values(j)[i] != '':
#                 uav_cluster_z[sheet_index].append(sheets[sheet_index].row_values(j)[i])
#
#         i = 3
#         for j in range(32, sheets[sheet_index].nrows):
#             if sheets[sheet_index].row_values(j)[i] != '':
#                 time_array[sheet_index].append(sheets[sheet_index].row_values(j)[i])
#
#         i = 4
#         if sheet_index == 0 or sheet_index == 1:
#             for j in range(32, sheets[sheet_index].nrows):
#                 if sheets[sheet_index].row_values(j)[i] != '':
#                     target_slice_array[sheet_index].append(float(sheets[sheet_index].row_values(j)[i].strip("[]")) - 0.5)
#                     # target_slice_array[sheet_index].append(0)
#         else:
#             for j in range(32, sheets[sheet_index].nrows):
#                 if sheets[sheet_index].row_values(j)[i] != '':
#                     target_slice_array[sheet_index].append(float(sheets[sheet_index].row_values(j)[i].strip("[]")))
#
#         ep_reward.append(sheets[sheet_index].row_values(1)[5])
#         episode.append(sheets[sheet_index].row_values(1)[6])
#         steps.append(sheets[sheet_index].row_values(1)[7])
#
#         ue_uav_bs_plot.plot(uav_cluster_x, uav_cluster_y, uav_cluster_z, ue_cluster, bs_cluster,
#                             time_array, target_slice_array, building_cluster, ep_reward, episode, steps)
