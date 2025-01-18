import numpy as np
from numpy.linalg import norm

def channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk):
    # Ensure no zero distances
    distances_RU_UE = np.maximum(distances_RU_UE, 1e-3)  # Minimum distance = 1 mm

    # Calculate path loss in dB and convert to linear scale
    path_loss_db = path_loss_ref + path_loss_exp * np.log10(distances_RU_UE / 1000)  # Distance in km
    path_loss_linear = 10 ** (-path_loss_db / 10)

    # Initialize channel power matrix
    channel_matrix = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))

    # Calculate channel power for each slice, RU, RB, and UE
    for s in range(num_slices):
        for i in range(num_RUs):
            for k in range(num_UEs):
                for b in range(num_RBs):
                    # Get channel gain from path loss
                    channel_gain = path_loss_linear[i, k]
                    # Generate Rayleigh fading channel
                    h = np.sqrt(path_loss_linear[i, k]) * np.sqrt(1 / 2) * (np.random.randn(num_antennas) + 1j * np.random.randn(num_antennas))
                    # Calculate channel power (norm squared)
                    channel_matrix[i, b, s, k] = norm(h, 2) ** 2

    # Calculate gain
    gain = channel_matrix / noise_power_density

    # Calculate received power
    received_power = channel_matrix * P_ib_sk 

    # Calculate SNR
    SNR = received_power / noise_power_density

    # Calculate data rate using Shannon formula
    data_rate = bandwidth_per_RB * np.log2(1 + SNR)

    return gain, SNR, data_rate

