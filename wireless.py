import numpy as np
from numpy.linalg import norm

def channel_gain(distances_RU_UE, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts):
# Tính suy hao đường truyền (Path Loss) dưới dạng dB
    path_loss_db = path_loss_ref + path_loss_exp * np.log10(distances_RU_UE / 1000)
    path_loss_linear = 10**(-path_loss_db / 10)
    # Tạo ma trận channel gain
    gain = np.zeros((num_RUs, num_UEs, num_RBs))

    # Tính độ lợi kênh cho mỗi cặp RU-UE
    for i in range(num_RUs):
        for k in range(num_UEs):
            for b in range(num_RBs):
                # Tính độ lợi kênh với Rayleigh fading và num_antennas
                channel_gain = path_loss_linear[i, k] / noise_power_watts
                h = np.sqrt(channel_gain) * np.sqrt(1 / 2) * (np.random.randn(num_antennas) + 1j * np.random.randn(num_antennas))
                
                # Lưu độ lợi kênh vào ma trận
                gain[i, k, b] = norm(h, 2)**2
    #print("gain: ", gain)
    return gain


def allocate_power(num_RUs, num_UEs, num_RBs, max_tx_power_mwatts):
    
    # Khởi tạo ma trận công suất
    p_bi_sk = np.zeros((num_RUs, num_RBs))

    # Tính toán công suất tối đa phân bổ cho mỗi RB
    pp = max_tx_power_mwatts / num_RBs

    # Phân bổ công suất cho từng RU cho dịch vụ
    for i in range(num_RUs):
        for b in range(num_RBs):
            p_bi_sk[i, b] = pp
    return p_bi_sk