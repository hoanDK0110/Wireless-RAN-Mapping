def print_mapping(pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    if pi_sk is not None:
        print("Giá trị của biến pi_sk:")
        print(pi_sk.value)  # In giá trị của biến pi_sk

        print("\nGiá trị của biến z_bi_sk:")
        for key, var in z_bi_sk.items():
            print(f"z_bi_sk[{key}] = {var.value}")  # In giá trị của biến z_bi_sk

        print("\nGiá trị của biến phi_i_sk:")
        print(phi_i_sk.value)  # In giá trị của biến phi_i_sk

        print("\nGiá trị của biến phi_j_sk:")
        print(phi_j_sk.value)  # In giá trị của biến phi_j_sk

        print("\nGiá trị của biến phi_m_sk:")
        print(phi_m_sk.value)  # In giá trị của biến phi_m_sk
    else:
        print("Không thể giải bài toán tối ưu.")