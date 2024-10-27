import numpy as np
from numpy.linalg import norm

def channel_gain(distances_RU_UE, num_RUs, num_UEs, num_RBs, noise_power_watts, num_antennas, path_loss_ref, path_loss_exp):
    gain = np.zeros((num_RUs, num_UEs, num_RBs), dtype=np.float64)
    
    # Tính mất mát đường truyền (path loss) cho mỗi RU-UE
    path_loss_db = path_loss_ref + path_loss_exp * np.log10(distances_RU_UE / 1000)
    path_loss_linear = 10 ** (-path_loss_db / 10)
    
    # Tính độ lợi kênh cho mỗi RU, UE và RB
    for i in range(num_RUs):  
        for k in range(num_UEs):  
            channel_gain = path_loss_linear[i, k] / noise_power_watts
            for b in range(num_RBs):  
                # Ma trận kênh truyền Rayleigh fading
                h = np.sqrt(channel_gain) * np.sqrt(1/2) * (np.random.randn(num_antennas) + 1j * np.random.randn(num_antennas))
                gain[i, k, b] = np.linalg.norm(h, 2) ** 2  # Tính norm của h và lấy bình phương
    return gain


def allocate_power(num_RUs, num_UEs, num_RBs, max_tx_power_watts):

    # Khởi tạo ma trận công suất
    p_bi_sk = np.zeros((num_RUs, num_UEs, num_RBs))

    # Tính toán công suất tối đa phân bổ cho mỗi RB
    pp = max_tx_power_watts / (num_UEs * num_RBs)

    # Phân bổ công suất cho từng RU cho dịch vụ eMBB
    for i in range(num_RUs):
        p_bi_sk[i, :, :] = pp * np.ones((num_UEs, num_RBs))

    return p_bi_sk