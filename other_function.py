import numpy as np
import pickle as pkl
import gzip as gz
import datetime
import time

def extract_optimization_results(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs,pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    """
    Trích xuất giá trị của tất cả các biến tối ưu (cvxpy.Variable) sau khi giải quyết bài toán.

    Args:
    - pi_sk (cvxpy.Variable): Ma trận pi_sk (boolean), với kích thước (num_slices, num_UEs).
    - z_ib_sk (np.array): Ma trận z_ib_sk (boolean), với kích thước (num_RUs, num_RBs, num_slices, num_UEs).
    - p_ib_sk (np.array): Ma trận p_ib_sk (liên tục), với kích thước (num_RUs, num_RBs, num_slices, num_UEs).
    - mu_ib_sk (np.array): Ma trận mu_ib_sk (liên tục), với kích thước (num_RUs, num_RBs, num_slices, num_UEs).
    - phi_i_sk (np.array): Ma trận phi_i_sk (liên tục), với kích thước (num_RUs, num_slices, num_UEs).
    - phi_j_sk (np.array): Ma trận phi_j_sk (liên tục), với kích thước (num_RUs, num_slices, num_UEs).
    - phi_m_sk (np.array): Ma trận phi_m_sk (liên tục), với kích thước (num_RUs, num_RBs, num_slices).

    Returns:
    - dict: Một từ điển chứa các mảng kết quả của tất cả các biến tối ưu.
    """

    arr_pi_sk = np.empty((num_slices, num_UEs), dtype=int) 
    for s in range(num_slices):
        for k in range(num_UEs):
            arr_pi_sk[s, k] = pi_sk[s, k].value

    # Extract z_ib_sk (binary)
    arr_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=int)
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    arr_z_ib_sk[i, b, s, k] = z_ib_sk[i, b, s, k].value

    # Extract p_ib_sk (continuous)
    arr_p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=float)
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    p_ib_sk[i, b, s, k] = p_ib_sk[i, b, s, k].value

    # Extract mu_ib_sk (continuous)
    arr_mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=float)
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    arr_mu_ib_sk[i, b, s, k] = mu_ib_sk[i, b, s, k].value

    # Extract phi_i_sk
    arr_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=int)
    for i in range(num_RUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                arr_phi_i_sk[i, s, k] = phi_i_sk[i, s, k].value

    # Extract phi_j_sk 
    arr_phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=int)
    for j in range(num_DUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                arr_phi_j_sk[j, s, k] = phi_j_sk[j, s, k].value

    # Extract phi_m_sk 
    arr_phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=int)
    for m in range(num_CUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                arr_phi_m_sk[m, s, k] = phi_m_sk[m, s, k].value

    return arr_pi_sk, arr_z_ib_sk, arr_p_ib_sk, arr_mu_ib_sk, arr_phi_i_sk, arr_phi_j_sk, arr_phi_m_sk

def save_object(filename, object):
    with gz.open(filename, mode="wb", compresslevel=9) as f:
        f.write(
            pkl.dumps(object)
        )
        
def load_object(filename):
    with gz.open(filename, mode='"rb') as f:
        return pkl.load(f)
    
class Stopwatch:
    def __init__(self, name, silent=True):
        self.name = name
        self.createat = datetime.datetime.now()
        self.startedat = None
        self.stopped = False
        self.checkpoints = []
        self.i = 0
        self.silent = silent
        
    def start(self):
        if self.stopped:
            raise Exception("Cannot start a stopped instance !!!")
        self.startedat = datetime.datetime.now()
        timestamp = round(time.process_time(), 5)
        r = (timestamp, "Started", 0)
        self.checkpoints.append(
            r
        )
        if not self.silent:
            print(f"StopWatch={self.name} | {self.__row2str(r)}")
        self.i += 1
    
    def add(self, contents=None):
        if self.stopped:
            raise Exception("Cannot add checkpoing to a stopped instance !!!")
        if self.startedat is None:
            raise Exception("Cannot add checkpoing to a not-yet-started instance !!!")
        if contents is None:
            contents = f"Checkpoint_{self.i}"
        timestamp = round(time.process_time(), 5)
        diff = round(timestamp - self.checkpoints[self.i-1][0], 5)
        r = (timestamp, contents, diff)
        self.checkpoints.append(
            r
        )
        self.i += 1
        if not self.silent:
            print(f"StopWatch={self.name} | {self.__row2str(r)}")
        
    def stop(self):
        if self.stopped:
            return
        if self.startedat is None:
            raise Exception("Cannot stop a not-yet-started instance !!!")
        self.stopped = True
        timestamp = round(time.process_time(), 5)
        diff = round(timestamp - self.checkpoints[self.i-1][0], 5)
        r = (timestamp, "Stopped", diff)
        self.checkpoints.append(
            r
        )
        self.i += 1
        if not self.silent:
            print(f"StopWatch={self.name} | {self.__row2str(r)}")

    def __repr__(self):
        return f"stopwatch name={self.name} created={self.createat.strftime('%Y%m%d_%H%M%S')} started={not (self.startedat is None)} stopped={self.stopped} n={self.i}"
    
    def __row2str(self, r):
        return f"{r[0]},{r[2]},{r[1]}"
        
    def write_to_file(self, filename):
        with open(filename, "wt") as f:
            f.write("time,diff,contents\n")
            for r in self.checkpoints:
                f.write(
                    self.__row2str(r)+"\n"
                )