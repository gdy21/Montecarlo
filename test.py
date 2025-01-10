import numpy as np

# 数据输入
eta = np.array([66.44, 66.54, 66.53, 66.46, 66.62, 66.53, 66.50, 66.59, 66.51])
R = np.array([70.91, 70.76, 70.83, 70.92, 70.81, 70.81, 70.98, 70.88, 70.89])

# 样本大小
N = len(eta)

# 计算均值
eta_mean = np.mean(eta)
R_mean = np.mean(R)

# 计算标准差
eta_std = np.sqrt(np.sum(eta**2) / N - (np.sum(eta) / N)**2)
R_std = np.sqrt(np.sum(R**2) / N - (np.sum(R) / N)**2)

# 置信水平 1 - alpha = 0.95，对应 lambda_a = 2.0
lambda_a = 2.0

# 计算相对误差
eta_error = (lambda_a * eta_std) / (np.sqrt(N) * eta_mean) * 100
R_error = (lambda_a * R_std) / (np.sqrt(N) * R_mean) * 100

# 输出结果
print(f'\u03B7 平均值: {eta_mean:.2f}%')
print(f'\u03B7 相对误差: {eta_error:.2f}%')
print(f'R 平均值: {R_mean:.2f}%')
print(f'R 相对误差: {R_error:.2f}%')
