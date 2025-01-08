import numpy as np
from numpy.linalg import norm

def channel_matrix(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts, rb_bandwidth, p_ib_sk):
    """
    Tính độ lợi kênh (Channel Gain) cho hệ thống MIMO.

    Parameters:
        distances_RU_UE (ndarray): Ma trận khoảng cách (num_RUs x num_UEs).
        num_slices (int): Số lượng lát cắt (slices).
        num_RUs (int): Số lượng RUs.
        num_UEs (int): Số lượng UEs.
        num_RBs (int): Số lượng Resource Blocks.
        num_antennas (int): Số anten tại mỗi RU và UE.
        path_loss_ref (float): Suy hao đường truyền tại khoảng cách tham chiếu (dB).
        path_loss_exp (float): Hệ số suy hao đường truyền.
        noise_power_watts (float): Công suất nhiễu (W).

    Returns:
        gain (ndarray): Ma trận độ lợi kênh (num_RUs x num_RBs x num_slices x num_UEs).
    """
    # Tính suy hao đường truyền (Path Loss) dưới dạng dB
    # Tính độ lợi kênh (Channel Power)
    path_loss_db = path_loss_ref + path_loss_exp * np.log10(distances_RU_UE / 1000)
    path_loss_linear = 10 ** (-path_loss_db / 10)

    channel_power = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))

    for s in range(num_slices):
        for i in range(num_RUs):
            for k in range(num_UEs):
                for b in range(num_RBs):
                    # Tính channel gain
                    channel_gain = path_loss_linear[i, k] / noise_power_watts
                    h = np.sqrt(channel_gain) * (
                        np.random.randn(num_antennas, num_antennas) +
                        1j * np.random.randn(num_antennas, num_antennas)
                    ) / np.sqrt(2)

                    # Frobenius norm
                    channel_power[i, b, s, k] = np.linalg.norm(h, 'fro') ** 2

    # Tính SNR
    received_power = channel_power * p_ib_sk
    SNR = received_power / noise_power_watts

    # Tính Data Rate
    data_rate = rb_bandwidth * np.log2(1 + SNR)

    return channel_power, SNR, data_rate

def channel_gain_1(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts):
# Tính suy hao đường truyền (Path Loss) dưới dạng dB
    path_loss_db = path_loss_ref + path_loss_exp * np.log10(distances_RU_UE / 1000)
    path_loss_linear = 10**(-path_loss_db / 10)
    # Tạo ma trận channel gain
    gain = np.zeros((num_RUs, num_RBs, num_UEs))

    # Tính độ lợi kênh cho mỗi cặp RU-UE
    for i in range(num_RUs):
        for k in range(num_UEs):
            for b in range(num_RBs):
                # Tính độ lợi kênh với Rayleigh fading và num_antennas
                channel_gain = path_loss_linear[i, k] / noise_power_watts
                h = np.sqrt(channel_gain) * np.sqrt(1 / 2) * (np.random.randn(num_antennas) + 1j * np.random.randn(num_antennas))
                
                # Lưu độ lợi kênh vào ma trận
                gain[i, b, k] = norm(h, 2)**2
    #print("gain: ", gain)
    return gain