def print_results(running_mode,num_slice,num_RUs,num_DUs,num_CUs,num_UEs,pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk):
    print("Kết quả tối ưu hóa:")
    
    # In ra số lượng UE được chấp nhận
    for s in range(num_slice):
        print(f"Giá trị của pi_sk tại slice {s} (UE được chấp nhận):")
        for k in range(num_UEs):
            print(pi_sk[s,k].value)
    
    # In ra phân bổ RB cho mỗi RU-UE
    print("\nGiá trị của z_bi_sk (phân bổ RB cho mỗi RU-UE):")
    for s in range(z_bi_sk.shape[0]):
        for i in range(z_bi_sk.shape[1]):
            for k in range(z_bi_sk.shape[2]):
                for b in range(z_bi_sk.shape[3]):
                    print(f"z_bi_sk[{s}, {i}, {k}, {b}] = {z_bi_sk[s, i, k, b].value}")

    # In ra mối liên hệ giữa RU và UE
    if running_mode == 0:
        for s in range(num_slice):
            print(f"\nGiá trị của phi_i_sk tại slice {s} (mối liên hệ giữa RU và UE):")
            for i in range(num_RUs):
                for k in range(num_UEs):
                    print(phi_i_sk[s,i,k].value)

    # In ra mối liên hệ giữa DU và UE
    for s in range(num_slice):
        print(f"\nGiá trị của phi_j_sk tại slice {s} (mối liên hệ giữa DU và UE):")
        for j in range(num_DUs):
            for k in range(num_UEs):
                print(phi_j_sk[s,j,k].value)

    # In ra mối liên hệ giữa CU và UE
    for s in range(num_slice):
        print(f"\nGiá trị của phi_m_sk tại slice {s} (mối liên hệ giữa CU và UE):")
        for m in range(num_CUs):
            for k in range(num_UEs):
                print(phi_m_sk[s,m,k].value)

    # In ra công suất phân bổ cho mỗi RB của RU-UE
    print("\nGiá trị của P_bi_sk (công suất phân bổ cho mỗi RB của RU-UE):")
    for s in range(P_bi_sk.shape[0]):   
        for i in range(P_bi_sk.shape[1]):
            for k in range(P_bi_sk.shape[2]):
                for b in range(P_bi_sk.shape[3]):
                    print(f"P_bi_sk[{s}, {i}, {k}, {b}] = {P_bi_sk[s, i, k, b].value}")

    # In ra giá trị mu_bi_sk (biến liên tục mu cho mỗi RB của RU-UE)
    print("\nGiá trị của mu_bi_sk (biến liên tục mu cho mỗi RB của RU-UE):")
    for s in range(mu_bi_sk.shape[0]):
        for i in range(mu_bi_sk.shape[1]):
            for k in range(mu_bi_sk.shape[2]):
                for b in range(mu_bi_sk.shape[3]):
                    print(f"mu_bi_sk[{s}, {i}, {k}, {b}] = {mu_bi_sk[s, i, k, b].value}")


def print_short_term_results(num_slice,num_RUs,num_DUs,num_CUs,num_UEs,R_sk_short_term,mu_bi_sk_short_term,z_bi_sk_short_term):
    print("\nGiá trị của R_sk_short_term:")
    print(R_sk_short_term.value)
    print("\nGiá trị của mu_bi_sk_short_term:")
    for s in range(mu_bi_sk_short_term.shape[0]):
        for i in range(mu_bi_sk_short_term.shape[1]):
            for k in range(mu_bi_sk_short_term.shape[2]):
                for b in range(mu_bi_sk_short_term.shape[3]):
                    print(f"mu_bi_sk_short_term[{s}, {i}, {k}, {b}] = {mu_bi_sk_short_term[s, i, k, b].value}")
    print("\nGiá trị của z_bi_sk :")
    for s in range(z_bi_sk_short_term.shape[0]):
        for i in range(z_bi_sk_short_term.shape[1]):
            for k in range(z_bi_sk_short_term.shape[2]):
                for b in range(z_bi_sk_short_term.shape[3]):
                    print(f"z_bi_sk_short_term[{s}, {i}, {k}, {b}] = {z_bi_sk_short_term[s, i, k, b].value}")