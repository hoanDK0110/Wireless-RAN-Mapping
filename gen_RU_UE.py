import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

def gen_coordinates_RU(num_RUs, radius):
    circle_RU_out = radius * 0.65
    angles = np.linspace(0, 2 * np.pi, num_RUs - 1, endpoint=False) 
    x = np.concatenate(([0], circle_RU_out * np.cos(angles)))  
    y = np.concatenate(([0], circle_RU_out * np.sin(angles)))  
    coordinates_RU = list(zip(x, y)) 
    return coordinates_RU

def gen_coordinates_UE(num_UEs, radius_in, radius_out):
    np.random.seed(1)
    angles = np.random.uniform(0, 2 * np.pi, num_UEs)
    r = np.random.uniform(radius_in, radius_out, num_UEs)
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    coordinates_UE = list(zip(x, y))  
    return coordinates_UE

def calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs):

    
    distances_RU_UE = np.zeros((num_RUs, num_UEs))
    
    
    for i in range(num_RUs):
        for j in range(num_UEs):
            x_RU, y_RU = coordinates_RU[i]
            x_UE, y_UE = coordinates_UE[j]
            distances_RU_UE[i, j] = np.sqrt((x_RU - x_UE)**2 + (y_RU - y_UE)**2)
    return distances_RU_UE


def gen_coordinates_UE_for_short_term(ue_index, coordinates_UE, k):
    x_current, y_current = coordinates_UE[ue_index]
    
    r = np.random.uniform(0, k)

    angle = np.random.uniform(0, 2 * np.pi)

    x_new = x_current + r * np.cos(angle)
    y_new = y_current + r * np.sin(angle)

    coordinates_UE[ue_index] = (x_new, y_new)
    return coordinates_UE


def plot_save_network(coordinates_RU, coordinates_UE, radius_in, radius_out):
    circle_in = plt.Circle((0, 0), radius_in, color='gray', fill=False, linestyle='--', label='Inner Radius')
    circle_out = plt.Circle((0, 0), radius_out, color='black', fill=False, linestyle='--', label='Outer Radius')
    
    plt.gca().add_artist(circle_in)
    plt.gca().add_artist(circle_out)
    
    for (x, y) in coordinates_RU:
        plt.scatter(x, y, color='green', marker='^', s=100, label='RU' if (x, y) == (0, 0) else "")
    
    for index, (x, y) in enumerate(coordinates_UE):
        plt.scatter(x, y, color='blue', marker='o')
        if index == 0:  # Chỉ chú thích cho UE đầu tiên
            plt.scatter(x, y, color='blue', marker='o', label='UE')

    plt.xlim(-radius_out * 1.2, radius_out * 1.2)
    plt.ylim(-radius_out * 1.2, radius_out * 1.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.title("Network with RU and UE")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right')
    
    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    fig_name = os.path.join(result_dir, f"network_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    plt.savefig(fig_name)
    plt.show(block=False)
    plt.pause(5)
    plt.close()