import numpy as np  

def print_results(pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk):
    print("Kết quả tối ưu hóa:")
    
    # In ra số lượng UE được chấp nhận
    print("Giá trị của pi_sk (UE được chấp nhận):")
    print(type(pi_sk)) 
    print(pi_sk.value)
    
    # In ra phân bổ RB cho mỗi RU-UE
    print("\nGiá trị của z_bi_sk (phân bổ RB cho mỗi RU-UE):")
    for i in range(z_bi_sk.shape[0]):
        for k in range(z_bi_sk.shape[1]):
            for b in range(z_bi_sk.shape[2]):
                print(f"z_bi_sk[{i}, {k}, {b}] = {z_bi_sk[i, k, b].value}")

    # In ra mối liên hệ giữa RU và UE
    print("\nGiá trị của phi_i_sk (mối liên hệ giữa RU và UE):")
    print(phi_i_sk.value)

    # In ra mối liên hệ giữa DU và UE
    print("\nGiá trị của phi_j_sk (mối liên hệ giữa DU và UE):")
    print(phi_j_sk.value)

    # In ra mối liên hệ giữa CU và UE
    print("\nGiá trị của phi_m_sk (mối liên hệ giữa CU và UE):")
    print(phi_m_sk.value)

    # In ra công suất phân bổ cho mỗi RB của RU-UE
    print("\nGiá trị của P_bi_sk (công suất phân bổ cho mỗi RB của RU-UE):")
    for i in range(P_bi_sk.shape[0]):
        for k in range(P_bi_sk.shape[1]):
            for b in range(P_bi_sk.shape[2]):
                print(f"P_bi_sk[{i}, {k}, {b}] = {P_bi_sk[i, k, b].value}")

    # In ra giá trị mu_bi_sk (biến liên tục mu cho mỗi RB của RU-UE)
    print("\nGiá trị của mu_bi_sk (biến liên tục mu cho mỗi RB của RU-UE):")
    for i in range(mu_bi_sk.shape[0]):
        for k in range(mu_bi_sk.shape[1]):
            for b in range(mu_bi_sk.shape[2]):
                print(f"mu_bi_sk[{i}, {k}, {b}] = {mu_bi_sk[i, k, b].value}")


def print_short_term_results(R_sk_short_term):
    print("Giá trị của R_sk:")
    if isinstance(R_sk_short_term, np.ndarray):
        for k in range(R_sk_short_term.shape[0]):
            print(f"R_sk[{k}] = {R_sk_short_term[k]}")
    else:
        print(R_sk_short_term)