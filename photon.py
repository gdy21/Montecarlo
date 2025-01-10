import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import sqrt, log
from multiprocessing import Pool, cpu_count

# 电子静止能量
E_electron = 0.511  # MeV

#---------------------------------------------
# 参数设置
#---------------------------------------------
N = int(1e6)           # 模拟的光子数
E_0 = 0.662            # 源能量(MeV)
R = 2.0               # NaI晶体半径(cm)
H = 8.0               # NaI晶体高度(cm)
D = 20.0              # 源到晶体上表面中心的距离(cm)
Al_thickness = 0.2   # 铝层厚度(cm)

# 能谱统计范围与分辨率
E_max = 0.85
dE = 0.002
bins = np.arange(0, E_max+dE, dE)
spectrum = np.zeros(len(bins) - 1)  # 能量计数器，长度为 len(bins) - 1

#---------------------------------------------
# 简化物理模型假设的函数
#---------------------------------------------

def load_cross_section_data(filename):
    """
    从文件中读取能量和截面数据。

    参数:
        filename (str): 包含截面数据的文件名。

    返回:
        dict: 包含能量、康普顿效应截面和光电效应截面的字典。
    """
    data = {
        'energies': [],
        'compton_sections': [],
        'photoelectric_sections': []
    }

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 跳过前两行标题，解析数据
    for line in lines[3:]:
        if line.strip():
            cols = line.split()
            if len(cols) == 3:
                try:
                    data['energies'].append(float(cols[0]))
                    data['compton_sections'].append(float(cols[1]))
                    data['photoelectric_sections'].append(float(cols[2]))
                except ValueError:
                    continue  # 跳过无法解析的行

    # 转换为NumPy数组
    data['energies'] = np.array(data['energies'])
    data['compton_sections'] = np.array(data['compton_sections'])
    data['photoelectric_sections'] = np.array(data['photoelectric_sections'])

    return data

def interpolate_cross_sections(energy, data):
    """
    使用线性插值法计算给定能量的康普顿效应截面和光电效应截面。

    参数:
        energy (float): 输入能量 (MeV)。
        data (dict): 包含能量和截面数据的字典。

    返回:
        (float, float): 插值计算的康普顿效应截面和光电效应截面。
    """
    energies = data['energies']
    compton_sections = data['compton_sections']
    photoelectric_sections = data['photoelectric_sections']

    # 检查能量范围
    if energy < energies[0] or energy > energies[-1]:
        raise ValueError(f"输入能量 {energy} 超出数据范围 ({energies[0]} - {energies[-1]} MeV)")

    # 使用线性插值
    compton_interp = np.interp(energy, energies, compton_sections)
    photoelectric_interp = np.interp(energy, energies, photoelectric_sections)

    cross_sections = np.array([
        compton_interp,  # 康普顿效应截面
        photoelectric_interp  # 光电效应截面
    ])

    # 计算反应概率
    total_cross_section = sum(cross_sections)
    probabilities = np.array([
        cross_sections[0] / total_cross_section,  # 康普顿效应概率
        cross_sections[1] / total_cross_section  # 光电效应概率
    ])

    return cross_sections, probabilities

# 读取文件数据（NaI.txt 和 Al.txt）
NaI_data = load_cross_section_data('NaI.txt')
Al_data = load_cross_section_data('Al.txt')


#---------------------------------------------
# 计算圆柱内一点沿方向角运动到圆柱侧边界的距离
#---------------------------------------------
def calculate_r_m(r_m_minus_1, varOmega_m_minus_1, R):
    """
    根据给定条件计算r_m

    参数:
    r_m_minus_1: 上一步的位置向量，形状为 (3,) 的numpy数组，代表 (x_{m-1}, y_{m-1}, z_{m-1})
    varOmega_m_minus_1: 上一步的方向向量，形状为 (3,) 的numpy数组，代表 (u_{m-1}, v_{m-1}, w_{m-1})
    R: 已知的半径值
    r_m: 计算得到的当前位置向量，形状为 (3,) 的numpy数组，代表 (x_m, y_m, z_m)
    x_m = x_m_minus_1 + L * u_m_minus_1
    y_m = y_m_minus_1 + L * v_m_minus_1
    z_m = z_m_minus_1 + L * w_m_minus_1
    返回值:
    圆柱内、外一点沿方向角运动到圆柱侧边界的距离
    """
    x_m_minus_1 = r_m_minus_1[0]
    y_m_minus_1 = r_m_minus_1[1]

    u_m_minus_1 = varOmega_m_minus_1[0]
    v_m_minus_1 = varOmega_m_minus_1[1]

    # 构建关于L的一元二次方程的系数
    a = u_m_minus_1 ** 2 + v_m_minus_1 ** 2
    b = 2 * (x_m_minus_1 * u_m_minus_1 + y_m_minus_1 * v_m_minus_1)
    c = x_m_minus_1 ** 2 + y_m_minus_1 ** 2 - R ** 2

    # 计算判别式
    discriminant = b ** 2 - 4 * a * c
    # 判别式小于0，则对应圆柱外一点因为方向角的缘故不可以运动到圆柱侧面，此时不可能存在材料跨越，L取一个很大的值
    if discriminant < 0:
        L=1e3
    elif discriminant == 0:
        L = -b / (2 * a)
    else:
        #这里只返回大的值，圆柱内一点负值表示往相反方向去了，圆柱外一点两个都是负值但是绝对值更大那个被舍弃了
        L = (-b + sqrt(discriminant)) / (2 * a)

    return L


def L_calculate(energy,r_value,varOmega):
    """
    计算多层介质的自由程 L从而获得下一个碰撞点位置
    :param xi: 随机数，介于 0 和 1 之间
    :param energy: 光子的能量 (可以在计算中用于确定吸收截面)
    :param r: 粒子的位置信息(3,)数组，xyz方向
    :param varOmega: 粒子的方向角信息(3,)数组，xyz方向
    :return: 下一个碰撞点位置
    """
    NaI_interp,_ = interpolate_cross_sections(energy, NaI_data)
    Al_interp,_ = interpolate_cross_sections(energy, Al_data)
    xi=np.random.rand()
    pho=-np.log(xi)
    if Al_thickness==0:
        # 只会在NaI层
        L = pho/sum(NaI_interp)
        r_next=r_value+L*varOmega
    else:
        #粒子不是沿着z轴方向，存在从NaI层跨到Al层的可能
        if np.linalg.norm(varOmega[:2])>0 and np.linalg.norm(r_value[:2])<=R: 
            L_limit = calculate_r_m(r_value,varOmega, R)
            if pho<=L_limit*sum(NaI_interp):
                # NaI层
                L = pho/sum(NaI_interp)
                r_next=r_value+L*varOmega
            else:
                # 从NaI层跨到Al层
                L = L_limit+(pho-L_limit*sum(NaI_interp))/sum(Al_interp)
                r_next=r_value+L*varOmega
        #粒子不是沿着z轴方向，存在从Al层跨到NaI层的可能
        elif np.linalg.norm(varOmega[:2])>0 and np.linalg.norm(r_value[:2])>R: 
            L_limit = -calculate_r_m(r_value,varOmega, R)
            if pho<=L_limit*sum(NaI_interp):
                # Al层
                L = pho/sum(Al_interp)
                r_next=r_value+L*varOmega
            else:
                # 从NaI层跨到Al层
                L = L_limit+(pho-L_limit*sum(Al_interp))/sum(NaI_interp)
                r_next=r_value+L*varOmega
        #粒子沿着z轴方向，不存在跨域材料的情况，根据区域限制选择合适的截面常数即可
        elif np.linalg.norm(r_value[:2])<=R and np.linalg.norm(varOmega[:2])==0:
            # NaI层
            L = pho/sum(NaI_interp)
            r_next=r_value+L*varOmega
        #粒子沿着z轴方向，不存在跨域材料的情况，根据区域限制选择合适的截面常数即可
        elif np.linalg.norm(r_value[:2])>R and np.linalg.norm(varOmega[:2])==0:
            # Al层
            L = pho/sum(Al_interp)
            r_next=r_value+L*varOmega
    return r_next

def determine_reaction_type( energy, varOmega, material_data):
    """
    随机数方法
    目的：得到新的方向角、沉积能量，如果方向角为零则粒子随机游动历史终止。

    通过随机数抽样方法来确定光电效应或康普顿效应的反应类型。
    :param xi: 随机数，介于 0 和 1 之间
    :param energy: 光子的能量 (可以在计算中用于确定吸收截面)
    :param varOmega: 粒子的方向角信息(3,)数组，xyz方向
    :param material: 字符串，表示材料的类型 ('NaI' 或 'Al')
    :return: 反应类型，光电效应或康普顿效应
    """
    
    # 读取截面数据
    _,probabilities = interpolate_cross_sections(energy, material_data)
    
    # 随机数抽样
    xi=np.random.rand()
    # 确定反应类型
    if xi <= probabilities[1]:  # 光电效应概率对应 probabilities[1]
        Delta_energy = energy #光电效应后沉积的光子能量
        varOmega_next=np.array([0,0,0])  # 光电效应后光子方向角为零
        energy_next=0 #光电效应后光子能量为零
        return Delta_energy,varOmega_next,energy_next
    else:  # 剩余概率即为康普顿效应
        alpha_value=energy/E_electron
        while True:  # 持续尝试，直到满足条件返回值
            # 随机数抽样
            xi_1 = np.random.rand()
            xi_2 = np.random.rand()
            xi_3 = np.random.rand()

            if xi_1 <= 27 / (4 * alpha_value + 29):
                # 按照 x_1 的条件
                x_1 = (1 + 2 * alpha_value) / (1 + 2 * alpha_value * xi_2)
                if xi_3 <= 0.5 * (((alpha_value + 1 - x_1) / alpha_value) ** 2 + 1):
                    # 满足条件，计算 alpha_value_prime 和 energy_prime
                    alpha_value_prime = alpha_value / x_1
                    mu_L=1-1/alpha_value_prime+1/alpha_value
                    a=mu_L
                    b=sqrt(1-mu_L**2)
                    # 康普顿散射后的光子能量
                    energy_prime = alpha_value_prime * E_electron
                    #康普顿散射后沉积的光子能量
                    Delta_energy = energy - energy_prime
                    energy_next=energy_prime
                    break  # 满足条件，退出循环
            else:
                # 按照 x_2 的条件
                x_2 = 1 + 2 * alpha_value * xi_2
                if xi_3 <= 27 * (x_2 - 1) ** 2 / (4 * x_2 ** 3):
                    # 满足条件，计算 alpha_value_prime 和 energy_prime
                    alpha_value_prime = alpha_value / x_2
                    mu_L=1-1/alpha_value_prime+1/alpha_value
                    a=mu_L
                    b=sqrt(1-mu_L**2)
                    # 康普顿散射后的光子能量
                    energy_prime = alpha_value_prime * E_electron
                    #康普顿散射后沉积的光子能量
                    Delta_energy = energy - energy_prime
                    energy_next=energy_prime
                    break  # 满足条件，退出循环
        # 只有在if判断后才能生效的代码
        # 继续计算粒子新的方向角信息(3,)数组varOmega_next，xyz方向           
        varphi=np.random.uniform(0, 2 * np.pi)
        c=np.cos(varphi)
        d=np.sin(varphi)
        #康普顿散射后方向角计算
        varOmega_next=np.array([0,0,0])
        if np.linalg.norm(varOmega[:2])<0.01:
            varOmega_next=np.array([b*c,b*d,a*varOmega[2]])
        else:
            varOmega_next[0]=a*varOmega[0]+(-b*c*varOmega[0]*varOmega[2]+b*d*varOmega[1])/np.linalg.norm(varOmega[:2])
            varOmega_next[1]=a*varOmega[1]+(-b*c*varOmega[1]*varOmega[2]-b*d*varOmega[0])/np.linalg.norm(varOmega[:2])
            varOmega_next[2]=a*varOmega[2]+b*c*np.linalg.norm(varOmega[:2])
        return Delta_energy, varOmega_next,energy_next

#---------------------------------------------
# 高斯展宽函数
#---------------------------------------------
def apply_resolution(energy):
    """
    应用高斯分辨率对能量进行展宽。
    """
    if energy <= 0:
        return 0
    fwhm = 0.01 + 0.05 * sqrt(energy + 0.4 * energy ** 2)
    sigma = 0.4247 * fwhm
    return energy + sigma * np.random.normal()

def sample_isotropic_point_source(D, R, H):
    """
    实现各向同性点源抽样，返回源的位置 r0 和方向角 Omega0。

    参数:
        D (float): 源到晶体上表面中心的距离 (cm)
        R (float): NaI晶体半径 (cm)
        H (float): NaI晶体高度 (cm)

    返回:
        tuple: r0 (起始位置), Omega0 (方向单位矢量)
    """
    # 生成均匀随机数 ξ
    xi = np.random.rand()

    # 计算 w0 (z方向的方向余弦)
    w0 = (1 - D / np.sqrt(D**2 + R**2)) * xi - 1


    # 确保 w0 在 [-1, 0] 范围内
    if w0 > 0 or w0 < -1:
        raise ValueError("w0 超出合法范围，需要检查计算公式")

    # 计算方向单位矢量 Omega0
    sqrt_term = np.sqrt(1 - w0**2)  # 计算 x-y 平面分量
    Omega0 = np.array([sqrt_term, 0, w0])

    # 计算起始位置 r0
    x = D * sqrt_term / -w0
    r0 = np.array([x, 0, H])

    return r0, Omega0

#---------------------------------------------
# 修改模拟光子函数
#---------------------------------------------
def simulate_photon(index):
    """
    单个光子的蒙特卡罗模拟。
    参数:
        index (int): 光子索引（仅用于区分任务）。
    返回:
        tuple: 局部光子谱计数、探测到的计数（E > 0）、全能峰计数。
    """
    local_spectrum = np.zeros(len(bins) - 1)
    #初始化
    #r_value, varOmega = sample_isotropic_point_source(D, R, H)
    varOmega = np.array([0.0, 0.0, -1.0])  # 源方向单位矢量
    r_value = np.array([0.0, 0.0, H])  # γ光子进入探测器的位置
    energy = E_0
    Delta_energy = 0

    detected = 0
    full_energy_peak = 0

    while True:
        if np.linalg.norm(r_value[:2]) <= R:
            r_value = L_calculate(energy, r_value, varOmega)
            if np.linalg.norm(r_value[:2]) > (R + Al_thickness) or r_value[2] > H or r_value[2] < 0:
                # 应用高斯展宽记录测量能量
                if Delta_energy > 0:
                    detected += 1
                    recorded_energy = Delta_energy
                    bin_index = np.digitize(recorded_energy, bins) - 1
                    if 0 <= bin_index < len(local_spectrum):
                        local_spectrum[bin_index] += 1
                    if abs(recorded_energy - E_0) <= 3 * 0.4247 * (0.01 + 0.05 * sqrt(E_0 + 0.4 * E_0 ** 2)):
                        full_energy_peak += 1
                break
            else:
                delta_energy, varOmega, energy = determine_reaction_type(energy, varOmega, NaI_data)
                Delta_energy += apply_resolution(delta_energy)
                if energy <= 1e-3:
                    if Delta_energy > 0:
                        detected += 1
                        recorded_energy = Delta_energy
                        bin_index = np.digitize(recorded_energy, bins) - 1
                        if 0 <= bin_index < len(local_spectrum):
                            local_spectrum[bin_index] += 1
                        if abs(recorded_energy - E_0) <= 3 * 0.4247 * (0.01 + 0.05 * sqrt(E_0 + 0.4 * E_0 ** 2)):
                            full_energy_peak += 1
                    break
        elif np.linalg.norm(r_value[:2]) > R and np.linalg.norm(r_value[:2]) <= (R + Al_thickness):
            r_value = L_calculate(energy, r_value, varOmega)
            if np.linalg.norm(r_value[:2]) > (R + Al_thickness) or r_value[2] > H or r_value[2] < 0:
                # 应用高斯展宽记录测量能量
                if Delta_energy > 0:
                    detected += 1
                    recorded_energy = Delta_energy
                    bin_index = np.digitize(recorded_energy, bins) - 1
                    if 0 <= bin_index < len(local_spectrum):
                        local_spectrum[bin_index] += 1
                    if abs(recorded_energy - E_0) <= 3 * 0.4247 * (0.01 + 0.05 * sqrt(E_0 + 0.4 * E_0 ** 2)):
                        full_energy_peak += 1
                break
            else:
                delta_energy, varOmega, energy = determine_reaction_type(energy, varOmega, Al_data)
                Delta_energy += apply_resolution(delta_energy)
                if energy <= 1e-3:
                    if Delta_energy > 0:
                        detected += 1
                        recorded_energy = Delta_energy
                        bin_index = np.digitize(recorded_energy, bins) - 1
                        if 0 <= bin_index < len(local_spectrum):
                            local_spectrum[bin_index] += 1
                        if abs(recorded_energy - E_0) <= 3 * 0.4247 * (0.01 + 0.05 * sqrt(E_0 + 0.4 * E_0 ** 2)):
                            full_energy_peak += 1
                    break
    return local_spectrum, detected, full_energy_peak

# 计算能量分辨率
def calculate_energy_resolution(spectrum, bins):
    """
    计算能谱的能量分辨率。
    参数:
        spectrum (array): 能谱计数数组。
        bins (array): 能量 bin 的边界数组。
        peak_energy (float): 全能峰的峰位（MeV）。
    返回:
        float: 能量分辨率（百分比）。
    """
    # 找到峰位对应的索引
    peak_index = np.argmax(spectrum)
    peak_count = spectrum[peak_index]
    peak_position = 0.5 * (bins[peak_index] + bins[peak_index + 1])

    # 计算半高对应的计数值
    half_max = peak_count / 2

    # 从峰位向两侧查找半高位置
    left_index = peak_index
    while left_index > 0 and spectrum[left_index] > half_max:
        left_index -= 1
    right_index = peak_index
    while right_index < len(spectrum) - 1 and spectrum[right_index] > half_max:
        right_index += 1

    # 计算 FWHM（半高全宽）
    fwhm = bins[right_index] - bins[left_index]

    # 使用公式计算能量分辨率
    resolution = (fwhm / peak_position) * 100
    return resolution, fwhm, peak_position, left_index, right_index

# 在主程序中调用计算能量分辨率
if __name__ == "__main__":
    num_processes = cpu_count()  # 获取可用 CPU 核心数
    with Pool(processes=num_processes) as pool:
        results = pool.map(simulate_photon, range(N))

    # 汇总局部光子谱
    total_detected = 0
    total_full_energy_peak = 0
    for local_spectrum, detected, full_energy_peak in results:
        spectrum += local_spectrum
        total_detected += detected
        total_full_energy_peak += full_energy_peak

    # 计算探测效率和峰总比
    detection_efficiency = total_detected / N
    peak_to_total_ratio = total_full_energy_peak / total_detected if total_detected > 0 else 0

    # 计算能量分辨率
    energy_resolution, fwhm, peak_position , left_index, right_index= calculate_energy_resolution(spectrum, bins)

    # 打印结果
    print(f"探测总数: {total_detected:.6f} ")
    print(f"全能峰计数: {total_full_energy_peak:.6f} ")
    print(f"探测效率: {100*detection_efficiency:.3f}%")
    print(f"峰总比: {100*peak_to_total_ratio:.3f}%")
    print(f"全能峰的峰位: {peak_position:.3f} MeV, 半高全宽: {fwhm:.3f} MeV")
    print(f"662KeV处的能量分辨率: {energy_resolution:.3f}%")
    

    # 绘制能谱图
    plt.figure(figsize=(8, 5))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.bar(bin_centers, spectrum, width=(bins[1] - bins[0]), alpha=0.7, color='orange', edgecolor='orange')

    # 标注全能峰和 FWHM
    plt.axvline(peak_position, color='r', linestyle='--', label='Peak Position')
    plt.axvline(bins[left_index], color='g', linestyle='--', label='FWHM Left')
    plt.axvline(bins[right_index], color='g', linestyle='--', label='FWHM Right')

    # 添加标题和轴标签
    plt.title("Energy Deposition Spectrum", fontsize=16)
    plt.xlabel("Deposited Energy (MeV)", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
