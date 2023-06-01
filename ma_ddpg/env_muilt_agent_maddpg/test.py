import numpy as np

# array_1 = np.zeros((1, 2))
# b = [1, 2]
# array_2 = np.zeros((1, 2))
# array_1[0][1] = -1
# array_2 = array_1
# print(array_1)
# print(np.min(array_1))
# print(np.argmin(array_1, axis =None))
# location = np.unravel_index(np.argmin(array_1, axis =None), array_1.shape, order='C')
# print(location)
# print(location[1])

# print(array_1 + array_2)
# print(array_1)
# print(array_2)
# c = np.array(b) + 1
# np.array(c)
# print(array_1)
# a = [1, 2, 3, 4, 5]
# ratio = np.zeros((1, 5))
# for i in range(len(a)):
#     a1 = (a[i] / sum(a))
#     ratio[0][i] = a1
#
# print(ratio)

# a = [1, 2, 3, 4, 5]
#
# print(np.mean(a))

# ue_local_strategy = np.zeros(5)
# print(ue_local_strategy)
string_numbers = "[1.2, 2.3, 3, 4, 5]"  # 包含五个浮点数的字符串

# 去除方括号
cleaned_string = string_numbers.strip("[]")

# 拆分为单个数字的字符串列表
number_strings = cleaned_string.split(", ")

# 转换为浮点数列表
float_numbers = [float(num) for num in number_strings]

# 计算平均值
average = sum(float_numbers) / len(float_numbers)

# 输出结果
print("平均值:", average)
