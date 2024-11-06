def print_results_short_term(a_sk, z_bi_sk, phi_i_sk):
    # Kiểm tra và in kết quả của a_sk
    if a_sk is not None:
        print("Giá trị của a_sk (tối ưu hóa phân bổ cho mỗi UE):")
        print(a_sk.value)  # In mảng a_sk
    else:
        print("Không có giá trị hợp lệ cho a_sk.")
        
    # Kiểm tra và in kết quả của z_bi_sk
    if z_bi_sk is not None:
        print("\nGiá trị của z_bi_sk (phân bổ RB cho mỗi RU-UE):")
        for i in range(z_bi_sk.shape[0]):  # num_RUs
            for k in range(z_bi_sk.shape[1]):  # num_UEs
                for b in range(z_bi_sk.shape[2]):  # num_RBs
                    if z_bi_sk[i, k, b] is not None:
                        print(f"z_bi_sk[{i}, {k}, {b}] = {z_bi_sk[i, k, b].value}")
    else:
        print("Không có giá trị hợp lệ cho z_bi_sk.")
        
    # Kiểm tra và in kết quả của phi_i_sk
    if phi_i_sk is not None:
        print("\nGiá trị của phi_i_sk (mối liên hệ giữa RU và UE):")
        for i in range(phi_i_sk.shape[0]):  # num_RUs
            for k in range(phi_i_sk.shape[1]):  # num_UEs
                print(f"phi_i_sk[{i}, {k}] = {phi_i_sk[i, k].value}")
    else:
        print("Không có giá trị hợp lệ cho phi_i_sk.")

def print_result_long_term(pi_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    # In kết quả của pi_sk
    if pi_sk is not None:
        print("Giá trị của pi_sk (tối ưu phân bổ cho mỗi UE):")
        print(pi_sk.value)
    else:
        print("Không có giá trị hợp lệ cho pi_sk.")
        
    # In kết quả của phi_i_sk
    if phi_i_sk is not None:
        print("\nGiá trị của phi_i_sk (mối liên hệ giữa RU và UE):")
        for i in range(phi_i_sk.shape[0]):  # num_RUs
            for k in range(phi_i_sk.shape[1]):  # num_UEs
                print(f"phi_i_sk[{i}, {k}] = {phi_i_sk[i, k].value}")
    else:
        print("Không có giá trị hợp lệ cho phi_i_sk.")
        
    # In kết quả của phi_j_sk
    if phi_j_sk is not None:
        print("\nGiá trị của phi_j_sk (mối liên hệ giữa DU và UE):")
        for j in range(phi_j_sk.shape[0]):  # num_DUs
            for k in range(phi_j_sk.shape[1]):  # num_UEs
                print(f"phi_j_sk[{j}, {k}] = {phi_j_sk[j, k].value}")
    else:
        print("Không có giá trị hợp lệ cho phi_j_sk.")
        
    # In kết quả của phi_m_sk
    if phi_m_sk is not None:
        print("\nGiá trị của phi_m_sk (mối liên hệ giữa CU và UE):")
        for m in range(phi_m_sk.shape[0]):  # num_CUs
            for k in range(phi_m_sk.shape[1]):  # num_UEs
                print(f"phi_m_sk[{m}, {k}] = {phi_m_sk[m, k].value}")
    else:
        print("Không có giá trị hợp lệ cho phi_m_sk.")