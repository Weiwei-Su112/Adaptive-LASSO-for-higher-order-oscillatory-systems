from GeneralInteraction import *


def check_larger_than_limit(model: GeneralInteraction, ops_limit=1.0):
    act_mat = model.run()
    op = [model.phase_coherence(vec) for vec in act_mat.T]
    ops_avg = np.average(op)
    # print(ops_avg)
    if ops_avg <= ops_limit:
        return True
    else:
        return False


def calc_avg_prob(model_arr, trial_num):
    sum2 = 0
    sum3 = 0
    for model in model_arr:
        if np.sum(model.coupling_coef2 != 0):
            sum2 += np.sum(model.conn_mat2)
        if np.sum(model.coupling_coef3 != 0):
            sum3 += np.sum(model.conn_mat3)

    avg_conn2 = sum2 / trial_num
    avg_conn3 = sum3 / trial_num

    avg_prob2 = avg_conn2 / (model.n_nodes * (model.n_nodes - 1))
    avg_prob3 = avg_conn3 / (model.n_nodes * (model.n_nodes - 1) * (model.n_nodes - 2) / 2)
    return avg_prob2, avg_prob3


def calc_avg_prob2(model_arr, trial_num):
    sum2 = 0
    sum3 = 0
    for model in model_arr:
        p2, p3 = model.get_connectivity()
        sum2 += p2
        sum3 += p3

    avg_prob2 = sum2 / trial_num
    avg_prob3 = sum3 / trial_num
    return avg_prob2, avg_prob3


def make_dirs(p_):
    now_ = datetime.now()
    date = now_.strftime("%Y%m%d")
    tm_string = now_.strftime("%H%M%S")
    file_path1 = "Typedata/" + date + "/"
    os.makedirs(file_path1, exist_ok=True)

    file_path_p = "Typedata/" + date + "/" + tm_string + f"_p={str(p_)}"
    os.mkdir(file_path_p)
    return file_path_p


def merge_seed_array(original_arr, output_arr):
    assert len(original_arr) == len(output_arr), "Lengths of two arrays needs to be equal. "
    copy_arr = np.copy(original_arr)
    for i in range(len(output_arr)):
        if output_arr[i] != original_arr[i]:
            copy_arr[i] = output_arr[i]

    return copy_arr


def main_for_type_test(natfreq_seed_, conn_seed_, coup_seed_, noise_seed_, reduce_seed, K_or_T=1, trial_num=5,
                       pre_conn_2_lst=None, pre_conn_3_lst=None, pre_reduce_lst=None, p_lst_=None, mle_threshold=0.1):
    # 1. uni parameters
    if p_lst_ is None:
        p_lst_ = [0.05, 0.1, 0.15]
    scan_num = 1
    n_nodes = 12

    dt = 0.02
    noise_sth = 0.2
    st_fr = 1 / 9
    inf_l = 2 / 3
    # filter variable
    ops_limit = 0.4
    # ops_limit = 1
    ratio = 6.0
    all_connected = False

    outputs = []
    # file_path_base, file_path_p_lst = make_dirs_for_accu(p_lst_, K_or_T)
    real_2_lsts = []
    real_3_lsts = []
    real_reduce_lsts = []

    for ir in range(len(p_lst_)):
        p_ = p_lst_[ir]
        path = make_dirs(p_)

        # paths = file_path_p_lst[ir]

        # 2. variable (K or T) (1 is K, 2 is T)
        assert K_or_T == 1 or K_or_T == 2, "K_or_T can only be 1 or 2. 1 is K and 2 is T. "
        natfreq_arr = np.random.default_rng(natfreq_seed_).normal(loc=1, scale=0.1, size=n_nodes)
        noise_arr = np.random.default_rng(noise_seed_).integers(0, 1e10, size=3)

        real_2_lst = []
        real_3_lst = []
        real_reduce_lst = []

        success_mle_2_arr = np.array([])
        success_mle_ori_2_arr = np.array([])
        success_lasso_2_arr = np.array([])
        success_ada_2_arr = np.array([])
        success_mle_3_arr = np.array([])
        success_mle_ori_3_arr = np.array([])
        success_lasso_3_arr = np.array([])
        success_ada_3_arr = np.array([])
        success_mle_mix_arr = np.array([])
        success_mle_ori_mix_arr = np.array([])
        success_lasso_mix_arr = np.array([])
        success_ada_mix_arr = np.array([])

        if K_or_T == 1:
            T = 900
            K_arr = np.linspace(0.1, 0.14, num=scan_num)
            # K_arr = np.array([0.001])
            hori_arr = np.copy(K_arr)

            for i in range(scan_num):
                if pre_conn_2_lst is None:
                    pre_conn_2_arr = None
                else:
                    pre_conn_2_arr = pre_conn_2_lst[ir][i]

                if pre_conn_3_lst is None:
                    pre_conn_3_arr = None
                else:
                    pre_conn_3_arr = pre_conn_3_lst[ir][i]

                if pre_reduce_lst is None:
                    pre_reduce_arr = None
                else:
                    pre_reduce_arr = pre_reduce_lst[ir][i]

                model_lst, real_conn_2_arr, real_conn_3_arr, real_reduce_arr = \
                    create_task_type(conn_seed_=conn_seed_, coup_seed_=coup_seed_, noise_arr=noise_arr,
                                     natfreq_arr=natfreq_arr, pre_conn_2_arr=pre_conn_2_arr, trial_num=trial_num,
                                     K=K_arr[i], T=T, dt=dt, p_=p_, noise_sth=noise_sth, ratio=ratio,
                                     st_fr=st_fr, inf_l=inf_l, reduce_seed_=reduce_seed,
                                     ops_limit=ops_limit, all_connected=all_connected, pre_reduce_arr=pre_reduce_arr,
                                     pre_conn_3_arr=pre_conn_3_arr)

                # if np.any(real_conn_2_arr != 0):
                #     print("The outcome conn2_arr is", real_conn_2_arr)
                real_2_lst.append(real_conn_2_arr)
                # if np.any(real_conn_3_arr != 0):
                #     print("The outcome conn3_arr is", real_conn_3_arr)
                real_3_lst.append(real_conn_3_arr)
                # if np.any(real_reduce_arr != 0):
                #     print("The outcome reduce_arr is", real_reduce_arr)
                real_reduce_lst.append(real_reduce_arr)

                assert len(model_lst) == 3, "len of model_lst has to be 3, pairwise, 3-int, and mixture. "

                success_mle_2 = 0
                success_mle_ori_2 = 0
                success_lasso_2 = 0
                success_ada_2 = 0
                success_mle_3 = 0
                success_mle_ori_3 = 0
                success_lasso_3 = 0
                success_ada_3 = 0
                success_mle_mix = 0
                success_mle_ori_mix = 0
                success_lasso_mix = 0
                success_ada_mix = 0

                success_mle_2_pair = np.zeros(trial_num)
                success_mle_ori_2_pair = np.zeros(trial_num)
                success_lasso_2_pair = np.zeros(trial_num)
                success_ada_2_pair = np.zeros(trial_num)
                success_mle_3_pair = np.zeros(trial_num)
                success_mle_ori_3_pair = np.zeros(trial_num)
                success_lasso_3_pair = np.zeros(trial_num)
                success_ada_3_pair = np.zeros(trial_num)
                success_mle_mix_pair = np.zeros(trial_num)
                success_mle_ori_mix_pair = np.zeros(trial_num)
                success_lasso_mix_pair = np.zeros(trial_num)
                success_ada_mix_pair = np.zeros(trial_num)

                success_mle_2_3int = np.zeros(trial_num)
                success_mle_ori_2_3int = np.zeros(trial_num)
                success_lasso_2_3int = np.zeros(trial_num)
                success_ada_2_3int = np.zeros(trial_num)
                success_mle_3_3int = np.zeros(trial_num)
                success_mle_ori_3_3int = np.zeros(trial_num)
                success_lasso_3_3int = np.zeros(trial_num)
                success_ada_3_3int = np.zeros(trial_num)
                success_mle_mix_3int = np.zeros(trial_num)
                success_mle_ori_mix_3int = np.zeros(trial_num)
                success_lasso_mix_3int = np.zeros(trial_num)
                success_ada_mix_3int = np.zeros(trial_num)
                for j in range(trial_num):
                    mle_re_2, lasso_re_2, ada_re_2, mle_ori_re_2 = type_test_demo(model_lst[0][j], mle_ori=True,
                                                                                  mle_threshold=mle_threshold)
                    mle_re_3, lasso_re_3, ada_re_3, mle_ori_re_3 = type_test_demo(model_lst[1][j], mle_ori=True,
                                                                                  mle_threshold=mle_threshold)
                    mle_re_mix, lasso_re_mix, ada_re_mix, mle_ori_re_mix = type_test_demo(model_lst[2][j], mle_ori=True,
                                                                                  mle_threshold=mle_threshold)
                    if mle_re_2[0] == "pairwise":
                        success_mle_2 += 1
                    if mle_ori_re_2[0] == "pairwise":
                        success_mle_ori_2 += 1
                    if lasso_re_2[0] == "pairwise":
                        success_lasso_2 += 1
                    if ada_re_2[0] == "pairwise":
                        success_ada_2 += 1
                    if mle_re_3[0] == "3-int":
                        success_mle_3 += 1
                    if mle_ori_re_3[0] == "3-int":
                        success_mle_ori_3 += 1
                    if lasso_re_3[0] == "3-int":
                        success_lasso_3 += 1
                    if ada_re_3[0] == "3-int":
                        success_ada_3 += 1
                    if mle_re_mix[0] == "mix":
                        success_mle_mix += 1
                    if mle_ori_re_mix[0] == "mix":
                        success_mle_ori_mix += 1
                    if lasso_re_mix[0] == "mix":
                        success_lasso_mix += 1
                    if ada_re_mix[0] == "mix":
                        success_ada_mix += 1

                    success_mle_2_pair[j] = mle_re_2[1]
                    success_mle_ori_2_pair[j] = mle_ori_re_2[1]
                    success_lasso_2_pair[j] = lasso_re_2[1]
                    success_ada_2_pair[j] = ada_re_2[1]
                    success_mle_3_pair[j] = mle_re_3[1]
                    success_mle_ori_3_pair[j] = mle_ori_re_3[1]
                    success_lasso_3_pair[j] = lasso_re_3[1]
                    success_ada_3_pair[j] = ada_re_3[1]
                    success_mle_mix_pair[j] = mle_re_mix[1]
                    success_mle_ori_mix_pair[j] = mle_ori_re_mix[1]
                    success_lasso_mix_pair[j] = lasso_re_mix[1]
                    success_ada_mix_pair[j] = ada_re_mix[1]

                    success_mle_2_3int[j] = mle_re_2[2]
                    success_mle_ori_2_3int[j] = mle_ori_re_2[2]
                    success_lasso_2_3int[j] = lasso_re_2[2]
                    success_ada_2_3int[j] = ada_re_2[2]
                    success_mle_3_3int[j] = mle_re_3[2]
                    success_mle_ori_3_3int[j] = mle_ori_re_3[2]
                    success_lasso_3_3int[j] = lasso_re_3[2]
                    success_ada_3_3int[j] = ada_re_3[2]
                    success_mle_mix_3int[j] = mle_re_mix[2]
                    success_mle_ori_mix_3int[j] = mle_ori_re_mix[2]
                    success_lasso_mix_3int[j] = lasso_re_mix[2]
                    success_ada_mix_3int[j] = ada_re_mix[2]

                success_mle_2_arr = np.append(success_mle_2_arr, success_mle_2 / trial_num)
                success_mle_ori_2_arr = np.append(success_mle_ori_2_arr, success_mle_ori_2 / trial_num)
                success_lasso_2_arr = np.append(success_lasso_2_arr, success_lasso_2 / trial_num)
                success_ada_2_arr = np.append(success_ada_2_arr, success_ada_2 / trial_num)
                success_mle_3_arr = np.append(success_mle_3_arr, success_mle_3 / trial_num)
                success_mle_ori_3_arr = np.append(success_mle_ori_3_arr, success_mle_ori_3 / trial_num)
                success_lasso_3_arr = np.append(success_lasso_3_arr, success_lasso_3 / trial_num)
                success_ada_3_arr = np.append(success_ada_3_arr, success_ada_3 / trial_num)
                success_mle_mix_arr = np.append(success_mle_mix_arr, success_mle_mix / trial_num)
                success_mle_ori_mix_arr = np.append(success_mle_ori_mix_arr, success_mle_ori_mix / trial_num)
                success_lasso_mix_arr = np.append(success_lasso_mix_arr, success_lasso_mix / trial_num)
                success_ada_mix_arr = np.append(success_ada_mix_arr, success_ada_mix / trial_num)

                success_csv = path + f"/K={K_arr[i]}.csv"
                prob_csv = path + f"/K={K_arr[i]}_prob.csv"

                with open(success_csv, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    header = [f"p={p_}_K={K_arr[i]}", "Adaptive LASSO", "LASSO", "OLS", "OLS with threshold"]
                    writer.writerow(header)
                    first_row = ["Pure Pairwise", success_ada_2 / trial_num, success_lasso_2 / trial_num,
                                 success_mle_2 / trial_num, success_mle_ori_2 / trial_num]
                    sec_row = ["3-interaction", success_ada_3 / trial_num, success_lasso_3 / trial_num,
                               success_mle_3 / trial_num, success_mle_ori_3 / trial_num]
                    third_row = ["Mixture", success_ada_mix / trial_num, success_lasso_mix / trial_num,
                                 success_mle_mix / trial_num, success_mle_ori_mix / trial_num]
                    writer.writerow(first_row)
                    writer.writerow(sec_row)
                    writer.writerow(third_row)

                with open(prob_csv, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    header = [f"p={p_}_K={K_arr[i]}", "Adaptive LASSO", "LASSO", "OLS", "OLS with threshold", "Real"]
                    writer.writerow(header)
                    avg2_2, avg3_2 = calc_avg_prob2(model_lst[0], trial_num)
                    avg2_3, avg3_3 = calc_avg_prob2(model_lst[1], trial_num)
                    avg2_mix, avg3_mix = calc_avg_prob2(model_lst[2], trial_num)
                    first_row = ["Pure Pairwise - pairwise ratio", np.average(success_ada_2_pair),
                                 np.average(success_lasso_2_pair), np.average(success_mle_2_pair),
                                 np.average(success_mle_ori_2_pair), round(avg2_2, 4)]
                    sec_row = ["Pure Pairwise - 3-int ratio", np.average(success_ada_2_3int),
                               np.average(success_lasso_2_3int), np.average(success_mle_2_3int),
                               np.average(success_mle_ori_2_3int), round(avg3_2, 4)]
                    third_row = ["Pure 3-body - pairwise ratio", np.average(success_ada_3_pair),
                                 np.average(success_lasso_3_pair), np.average(success_mle_3_pair),
                                 np.average(success_mle_ori_3_pair), round(avg2_3, 4)]
                    fourth_row = ["Pure 3-body - 3-int ratio", np.average(success_ada_3_3int),
                                  np.average(success_lasso_3_3int), np.average(success_mle_3_3int),
                                  np.average(success_mle_ori_3_3int), round(avg3_3, 4)]
                    fifth_row = ["Mixture - pairwise ratio", np.average(success_ada_mix_pair),
                                 np.average(success_lasso_mix_pair), np.average(success_mle_mix_pair),
                                 np.average(success_mle_ori_mix_pair),
                                 round(avg2_mix, 4)]
                    sixth_row = ["Mixture - 3-int ratio", np.average(success_ada_mix_3int),
                                 np.average(success_lasso_mix_3int), np.average(success_mle_mix_3int),
                                 np.average(success_mle_ori_mix_3int),
                                 round(avg3_mix, 4)]
                    writer.writerow(first_row)
                    writer.writerow(sec_row)
                    writer.writerow(third_row)
                    writer.writerow(fourth_row)
                    writer.writerow(fifth_row)
                    writer.writerow(sixth_row)

        else:
            K = 0.1
            # T_arr = np.geomspace(20, 900, num=scan_num)
            # hori_arr = np.copy(T_arr) / (2 * np.pi)
            # hori_arr = inf_l * hori_arr
            # hori_arr = hori_arr.astype(int)

            hori_arr = np.array([2, 5, 10, 20, 50, 100, 200])
            T_arr = hori_arr * (2 * np.pi) / inf_l
            scan_num = len(T_arr)

            for i in range(scan_num):
                if pre_conn_2_lst is None:
                    pre_conn_2_arr = None
                else:
                    pre_conn_2_arr = pre_conn_2_lst[ir][i]

                if pre_conn_3_lst is None:
                    pre_conn_3_arr = None
                else:
                    pre_conn_3_arr = pre_conn_3_lst[ir][i]

                model_lst, real_conn_2_arr, real_conn_3_arr, real_reduce_arr = \
                    create_task_type(conn_seed_=conn_seed_, coup_seed_=coup_seed_, noise_arr=noise_arr,
                                     natfreq_arr=natfreq_arr, pre_conn_2_arr=pre_conn_2_arr, trial_num=trial_num, K=K,
                                     T=T_arr[i], dt=dt, p_=p_, noise_sth=noise_sth, ratio=ratio,
                                     st_fr=st_fr, inf_l=inf_l, reduce_seed_=reduce_seed, pre_reduce_arr=pre_reduce_arr,
                                     ops_limit=ops_limit, all_connected=all_connected, pre_conn_3_arr=pre_conn_3_arr)
                # if np.any(real_conn_2_arr != 0):
                #     print("The outcome conn2_arr is", real_conn_2_arr)
                real_2_lst.append(real_conn_2_arr)
                # if np.any(real_conn_3_arr != 0):
                #     print("The outcome conn3_arr is", real_conn_3_arr)
                real_3_lst.append(real_conn_3_arr)
                # if np.any(real_reduce_arr != 0):
                #     print("The outcome reduce_arr is", real_reduce_arr)
                real_reduce_lst.append(real_reduce_arr)

                success_mle_2 = 0
                success_mle_ori_2 = 0
                success_lasso_2 = 0
                success_ada_2 = 0
                success_mle_3 = 0
                success_mle_ori_3 = 0
                success_lasso_3 = 0
                success_ada_3 = 0
                success_mle_mix = 0
                success_mle_ori_mix = 0
                success_lasso_mix = 0
                success_ada_mix = 0

                success_mle_2_pair = np.zeros(trial_num)
                success_mle_ori_2_pair = np.zeros(trial_num)
                success_lasso_2_pair = np.zeros(trial_num)
                success_ada_2_pair = np.zeros(trial_num)
                success_mle_3_pair = np.zeros(trial_num)
                success_mle_ori_3_pair = np.zeros(trial_num)
                success_lasso_3_pair = np.zeros(trial_num)
                success_ada_3_pair = np.zeros(trial_num)
                success_mle_mix_pair = np.zeros(trial_num)
                success_mle_ori_mix_pair = np.zeros(trial_num)
                success_lasso_mix_pair = np.zeros(trial_num)
                success_ada_mix_pair = np.zeros(trial_num)

                success_mle_2_3int = np.zeros(trial_num)
                success_mle_ori_2_3int = np.zeros(trial_num)
                success_lasso_2_3int = np.zeros(trial_num)
                success_ada_2_3int = np.zeros(trial_num)
                success_mle_3_3int = np.zeros(trial_num)
                success_mle_ori_3_3int = np.zeros(trial_num)
                success_lasso_3_3int = np.zeros(trial_num)
                success_ada_3_3int = np.zeros(trial_num)
                success_mle_mix_3int = np.zeros(trial_num)
                success_mle_ori_mix_3int = np.zeros(trial_num)
                success_lasso_mix_3int = np.zeros(trial_num)
                success_ada_mix_3int = np.zeros(trial_num)
                for j in range(trial_num):
                    mle_re_2, lasso_re_2, ada_re_2, mle_ori_re_2 = type_test_demo(model_lst[0][j], mle_ori=True,
                                                                                  mle_threshold=mle_threshold)
                    mle_re_3, lasso_re_3, ada_re_3, mle_ori_re_3 = type_test_demo(model_lst[1][j], mle_ori=True,
                                                                                  mle_threshold=mle_threshold)
                    mle_re_mix, lasso_re_mix, ada_re_mix, mle_ori_re_mix = type_test_demo(model_lst[2][j], mle_ori=True,
                                                                                  mle_threshold=mle_threshold)
                    if mle_re_2[0] == "pairwise":
                        success_mle_2 += 1
                    if mle_ori_re_2[0] == "pairwise":
                        success_mle_ori_2 += 1
                    if lasso_re_2[0] == "pairwise":
                        success_lasso_2 += 1
                    if ada_re_2[0] == "pairwise":
                        success_ada_2 += 1
                    if mle_re_3[0] == "3-int":
                        success_mle_3 += 1
                    if mle_ori_re_3[0] == "3-int":
                        success_mle_ori_3 += 1
                    if lasso_re_3[0] == "3-int":
                        success_lasso_3 += 1
                    if ada_re_3[0] == "3-int":
                        success_ada_3 += 1
                    if mle_re_mix[0] == "mix":
                        success_mle_mix += 1
                    if mle_ori_re_mix[0] == "mix":
                        success_mle_ori_mix += 1
                    if lasso_re_mix[0] == "mix":
                        success_lasso_mix += 1
                    if ada_re_mix[0] == "mix":
                        success_ada_mix += 1

                    success_mle_2_pair[j] = mle_re_2[1]
                    success_mle_ori_2_pair[j] = mle_ori_re_2[1]
                    success_lasso_2_pair[j] = lasso_re_2[1]
                    success_ada_2_pair[j] = ada_re_2[1]
                    success_mle_3_pair[j] = mle_re_3[1]
                    success_mle_ori_3_pair[j] = mle_ori_re_3[1]
                    success_lasso_3_pair[j] = lasso_re_3[1]
                    success_ada_3_pair[j] = ada_re_3[1]
                    success_mle_mix_pair[j] = mle_re_mix[1]
                    success_mle_ori_mix_pair[j] = mle_ori_re_mix[1]
                    success_lasso_mix_pair[j] = lasso_re_mix[1]
                    success_ada_mix_pair[j] = ada_re_mix[1]

                    success_mle_2_3int[j] = mle_re_2[2]
                    success_mle_ori_2_3int[j] = mle_ori_re_2[2]
                    success_lasso_2_3int[j] = lasso_re_2[2]
                    success_ada_2_3int[j] = ada_re_2[2]
                    success_mle_3_3int[j] = mle_re_3[2]
                    success_mle_ori_3_3int[j] = mle_ori_re_3[2]
                    success_lasso_3_3int[j] = lasso_re_3[2]
                    success_ada_3_3int[j] = ada_re_3[2]
                    success_mle_mix_3int[j] = mle_re_mix[2]
                    success_mle_ori_mix_3int[j] = mle_ori_re_mix[2]
                    success_lasso_mix_3int[j] = lasso_re_mix[2]
                    success_ada_mix_3int[j] = ada_re_mix[2]

                success_mle_2_arr = np.append(success_mle_2_arr, success_mle_2 / trial_num)
                success_mle_ori_2_arr = np.append(success_mle_ori_2_arr, success_mle_ori_2 / trial_num)
                success_lasso_2_arr = np.append(success_lasso_2_arr, success_lasso_2 / trial_num)
                success_ada_2_arr = np.append(success_ada_2_arr, success_ada_2 / trial_num)
                success_mle_3_arr = np.append(success_mle_3_arr, success_mle_3 / trial_num)
                success_mle_ori_3_arr = np.append(success_mle_ori_3_arr, success_mle_ori_3 / trial_num)
                success_lasso_3_arr = np.append(success_lasso_3_arr, success_lasso_3 / trial_num)
                success_ada_3_arr = np.append(success_ada_3_arr, success_ada_3 / trial_num)
                success_mle_mix_arr = np.append(success_mle_mix_arr, success_mle_mix / trial_num)
                success_mle_ori_mix_arr = np.append(success_mle_ori_mix_arr, success_mle_ori_mix / trial_num)
                success_lasso_mix_arr = np.append(success_lasso_mix_arr, success_lasso_mix / trial_num)
                success_ada_mix_arr = np.append(success_ada_mix_arr, success_ada_mix / trial_num)

                success_csv = path + f"/T={T_arr[i]}.csv"
                prob_csv = path + f"/T={T_arr[i]}_prob.csv"

                with open(success_csv, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    header2 = [f"p={p_}_K={T_arr[i]}", "Adaptive LASSO", "LASSO", "OLS with FDR control",
                               "OLS with threshold"]
                    writer.writerow(header2)
                    first_row = ["Pairwise", success_ada_2 / trial_num, success_lasso_2 / trial_num,
                                 success_mle_2 / trial_num, success_mle_ori_2 / trial_num]
                    sec_row = ["3-interaction", success_ada_3 / trial_num, success_lasso_3 / trial_num,
                               success_mle_3 / trial_num, success_mle_ori_3 / trial_num]
                    third_row = ["Mixture", success_ada_mix / trial_num, success_lasso_mix / trial_num,
                                 success_mle_mix / trial_num, success_mle_ori_mix / trial_num]
                    writer.writerow(first_row)
                    writer.writerow(sec_row)
                    writer.writerow(third_row)

                with open(prob_csv, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    header = [f"p={p_}_K={K_arr[i]}", "Adaptive LASSO", "LASSO", "OLS with FDR control",
                              "OLS with threshold", "Real"]
                    writer.writerow(header)
                    avg2_2, avg3_2 = calc_avg_prob2(model_lst[0], trial_num)
                    avg2_3, avg3_3 = calc_avg_prob2(model_lst[1], trial_num)
                    avg2_mix, avg3_mix = calc_avg_prob2(model_lst[2], trial_num)
                    first_row = ["Pure Pairwise - pairwise ratio", np.average(success_ada_2_pair),
                                 np.average(success_lasso_2_pair), np.average(success_mle_2_pair),
                                 np.average(success_mle_ori_2_pair), round(avg2_2, 4)]
                    sec_row = ["Pure Pairwise - 3-int ratio", np.average(success_ada_2_3int),
                               np.average(success_lasso_2_3int), np.average(success_mle_2_3int),
                               np.average(success_mle_ori_2_3int), round(avg3_2, 4)]
                    third_row = ["Pure 3-body - pairwise ratio", np.average(success_ada_3_pair),
                                 np.average(success_lasso_3_pair), np.average(success_mle_3_pair),
                                 np.average(success_mle_ori_3_pair), round(avg2_3, 4)]
                    fourth_row = ["Pure 3-body - 3-int ratio", np.average(success_ada_3_3int),
                                  np.average(success_lasso_3_3int), np.average(success_mle_3_3int),
                                  np.average(success_mle_ori_3_3int), round(avg3_3, 4)]
                    fifth_row = ["Mixture - pairwise ratio", np.average(success_ada_mix_pair),
                                 np.average(success_lasso_mix_pair), np.average(success_mle_mix_pair),
                                 np.average(success_mle_ori_mix_pair),
                                 round(avg2_mix, 4)]
                    sixth_row = ["Mixture - 3-int ratio", np.average(success_ada_mix_3int),
                                 np.average(success_lasso_mix_3int), np.average(success_mle_mix_3int),
                                 np.average(success_mle_ori_mix_3int),
                                 round(avg3_mix, 4)]
                    writer.writerow(first_row)
                    writer.writerow(sec_row)
                    writer.writerow(third_row)
                    writer.writerow(fourth_row)
                    writer.writerow(fifth_row)
                    writer.writerow(sixth_row)

        output = {"x": hori_arr,
                  "ada": [success_ada_2_arr, success_ada_3_arr, success_ada_mix_arr],
                  "lasso": [success_lasso_2_arr, success_lasso_3_arr, success_lasso_mix_arr],
                  "mle": [success_mle_2_arr, success_mle_3_arr, success_mle_mix_arr],
                  "mle_ori": [success_mle_ori_2_arr, success_mle_ori_3_arr, success_mle_ori_mix_arr]}
        outputs.append(output)

        real_2_lsts.append(real_2_lst)
        real_3_lsts.append(real_3_lst)
        real_reduce_lsts.append(real_reduce_lst)

    np.save(f"{path}/new_two_seeds.npy", real_2_lsts)
    np.save(f"{path}/new_three_seeds.npy", real_3_lsts)
    np.save(f"{path}/new_reduce_seeds.npy", real_reduce_lsts)

    drawing_type_test(outputs, p_lst_, sharex=True, K_or_T=K_or_T)


def drawing_type_test(outputs, p_lst_, path_lst=None, path_base=None, sharex=False, K_or_T=1):
    assert len(outputs) == len(p_lst_), "check length of outputs and p list. "
    if sharex:
        hori_arr = outputs[0]["x"]

    if len(p_lst_) == 1:
        fig, ax = plt.subplots(nrows=len(p_lst_), ncols=3, sharex='col', sharey=True,
                               figsize=(18, 9), dpi=100)
    else:
        fig, ax = plt.subplots(nrows=len(p_lst_), ncols=3, sharex='col', sharey=True,
                               figsize=(18, int(6 * len(p_lst_) + 3)), dpi=100)
    if len(outputs) == 1:
        if not sharex:
            hori_arr = outputs[0]["x"]

        accu_arr_2_ada = outputs[0]["ada"][0]
        accu_arr_3_ada = outputs[0]["ada"][1]
        accu_arr_mix_ada = outputs[0]["ada"][2]

        accu_arr_2_lasso = outputs[0]["lasso"][0]
        accu_arr_3_lasso = outputs[0]["lasso"][1]
        accu_arr_mix_lasso = outputs[0]["lasso"][2]

        accu_arr_2_mle = outputs[0]["mle"][0]
        accu_arr_3_mle = outputs[0]["mle"][1]
        accu_arr_mix_mle = outputs[0]["mle"][2]

        accu_arr_2_mle_ori = outputs[0]["mle_ori"][0]
        accu_arr_3_mle_ori = outputs[0]["mle_ori"][1]
        accu_arr_mix_mle_ori = outputs[0]["mle_ori"][2]

        ax[0].plot(hori_arr, accu_arr_2_ada, label="Pairwise, A. LASSO", linewidth=10)
        ax[1].plot(hori_arr, accu_arr_3_ada, label="3-int, A. LASSO", linewidth=10)
        ax[2].plot(hori_arr, accu_arr_mix_ada, label="Mix, A. LASSO", linewidth=10)

        ax[0].plot(hori_arr, accu_arr_2_lasso, label="Pairwise, LASSO", linewidth=10, alpha=0.6)
        ax[1].plot(hori_arr, accu_arr_3_lasso, label="3-int, LASSO", linewidth=10, alpha=0.6)
        ax[2].plot(hori_arr, accu_arr_mix_lasso, label="Mix, LASSO", linewidth=10, alpha=0.6)

        ax[0].plot(hori_arr, accu_arr_2_mle, label="Pairwise, OLS with FDR control", linewidth=5, alpha=0.3)
        ax[1].plot(hori_arr, accu_arr_3_mle, label="3-int, OLS with FDR control", linewidth=5, alpha=0.3)
        ax[2].plot(hori_arr, accu_arr_mix_mle, label="Mix, OLS with FDR control", linewidth=5, alpha=0.3)

        ax[0].plot(hori_arr, accu_arr_2_mle_ori, label="Pairwise, OLS with threshold", linewidth=5, alpha=0.3)
        ax[1].plot(hori_arr, accu_arr_3_mle_ori, label="3-int, OLS with threshold", linewidth=5, alpha=0.3)
        ax[2].plot(hori_arr, accu_arr_mix_mle_ori, label="Mix, OLS with threshold", linewidth=5, alpha=0.3)

        ax[0].spines[['right', 'top']].set_visible(False)
        ax[1].spines[['right', 'top']].set_visible(False)
        ax[2].spines[['right', 'top']].set_visible(False)

        ax[0].tick_params('both', length=15, width=2, which='major')
        ax[0].tick_params('both', length=10, width=1, which='minor')
        ax[1].tick_params('both', length=15, width=2, which='major')
        ax[1].tick_params('both', length=10, width=1, which='minor')
        ax[2].tick_params('both', length=15, width=2, which='major')
        ax[2].tick_params('both', length=10, width=1, which='minor')

        ax[0].set_ylabel('Accuracy', rotation=90, fontsize=24)
        ax[0].tick_params(axis='y', labelsize=18)

        title_1 = f"Pairwise $p={p_lst_[0]}$"
        title_2 = f"3-int $p={p_lst_[0]}$"
        title_3 = f"Mix $p={p_lst_[0]}$"
        ax[0].set_title(title_1, fontsize=30)
        ax[1].set_title(title_2, fontsize=30)
        ax[2].set_title(title_3, fontsize=30)

        ax[0].set_ylim(-0.01, 1.01)

        # plt.rcParams["font.size"] = 36  # フォントを大きく
        plt.rcParams['font.family'] = 'Arial'
        params = {'legend.fontsize': 20,
                  'legend.handlelength': 1}
        plt.rcParams.update(params)
        fig.legend(ax, labels=["Adaptive LASSO", "LASSO", "OLS"], loc="lower right", framealpha=0.4)
        fig.suptitle('Z-test of type accuracy', size=40)
        if K_or_T == 1:
            xlabel = 'Coupling strength'
        else:
            xlabel = 'Cycle number'
            ax[0].set_xscale("log")
            ax[1].set_xscale("log")
            ax[2].set_xscale("log")

        ax[0].set_xlabel(xlabel, fontsize=24)
        ax[0].set_xticks(hori_arr)
        ax[0].tick_params(axis='x', labelsize=18)
        ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax[1].set_xlabel(xlabel, fontsize=24)
        ax[1].set_xticks(hori_arr)
        ax[1].tick_params(axis='x', labelsize=18)
        ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax[2].set_xlabel(xlabel, fontsize=24)
        ax[2].set_xticks(hori_arr)
        ax[2].tick_params(axis='x', labelsize=18)
        ax[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # ax.yaxis.set_label_coords(-0.08, 0.5)
        # ax.yaxis.set_major_formatter(formatter)

        # path = path_base + "MCC.jpg"
        # fig.savefig(path)

    else:
        for i in range(len(outputs)):
            if not sharex:
                hori_arr = outputs[i]["x"]

            accu_arr_2_ada = outputs[i]["ada"][0]
            accu_arr_3_ada = outputs[i]["ada"][1]
            accu_arr_mix_ada = outputs[i]["ada"][2]

            accu_arr_2_lasso = outputs[i]["lasso"][0]
            accu_arr_3_lasso = outputs[i]["lasso"][1]
            accu_arr_mix_lasso = outputs[i]["lasso"][2]

            accu_arr_2_mle = outputs[i]["mle"][0]
            accu_arr_3_mle = outputs[i]["mle"][1]
            accu_arr_mix_mle = outputs[i]["mle"][2]

            accu_arr_2_mle_ori = outputs[i]["mle_ori"][0]
            accu_arr_3_mle_ori = outputs[i]["mle_ori"][1]
            accu_arr_mix_mle_ori = outputs[i]["mle_ori"][2]

            ax[0, i].plot(hori_arr, accu_arr_2_ada, label="Pairwise, A. LASSO", linewidth=10)
            ax[1, i].plot(hori_arr, accu_arr_3_ada, label="3-int, A. LASSO", linewidth=10)
            ax[2, i].plot(hori_arr, accu_arr_mix_ada, label="Mix, A. LASSO", linewidth=10)

            ax[0, i].plot(hori_arr, accu_arr_2_lasso, label="Pairwise, LASSO", linewidth=10, alpha=0.6)
            ax[1, i].plot(hori_arr, accu_arr_3_lasso, label="3-int, LASSO", linewidth=10, alpha=0.6)
            ax[2, i].plot(hori_arr, accu_arr_mix_lasso, label="Mix, LASSO", linewidth=10, alpha=0.6)

            ax[0, i].plot(hori_arr, accu_arr_2_mle, label="Pairwise, OLS with FDR control", linewidth=5, alpha=0.3)
            ax[1, i].plot(hori_arr, accu_arr_3_mle, label="3-int, OLS with FDR control", linewidth=5, alpha=0.3)
            ax[2, i].plot(hori_arr, accu_arr_mix_mle, label="Mix, OLS with FDR control", linewidth=5, alpha=0.3)

            ax[0, i].plot(hori_arr, accu_arr_2_mle_ori, label="Pairwise, OLS with threshold", linewidth=5, alpha=0.3)
            ax[1, i].plot(hori_arr, accu_arr_3_mle_ori, label="3-int, OLS with threshold", linewidth=5, alpha=0.3)
            ax[2, i].plot(hori_arr, accu_arr_mix_mle_ori, label="Mix, OLS with threshold", linewidth=5, alpha=0.3)

            ax[0, i].spines[['right', 'top']].set_visible(False)
            ax[1, i].spines[['right', 'top']].set_visible(False)
            ax[2, i].spines[['right', 'top']].set_visible(False)

            ax[0, i].tick_params('both', length=15, width=2, which='major')
            ax[0, i].tick_params('both', length=10, width=1, which='minor')
            ax[1, i].tick_params('both', length=15, width=2, which='major')
            ax[1, i].tick_params('both', length=10, width=1, which='minor')
            ax[2, i].tick_params('both', length=15, width=2, which='major')
            ax[2, i].tick_params('both', length=10, width=1, which='minor')

            ax[0, i].set_ylabel('Accuracy', rotation=90, fontsize=24)
            ax[0, i].tick_params(axis='y', labelsize=18)

            title_1 = f"Pairwise $p={p_lst_[i]}$"
            title_2 = f"3-int $p={p_lst_[i]}$"
            title_3 = f"Mix $p={p_lst_[i]}$"
            ax[0, i].set_title(title_1, fontsize=30)
            ax[1, i].set_title(title_2, fontsize=30)
            ax[2, i].set_title(title_3, fontsize=30)

            ax[0, i].set_ylim(-0.01, 1.01)

            # paths = path_lst[i]

            # accu_txt_2_pth = paths + f"MCC_pairwise_p={p_lst_[i]}.csv"
            # accu_txt_3_pth = paths + f"MCC_3-interaction_p={p_lst_[i]}.csv"
            # accu_txt_mix_pth = paths + f"MCC_mixture_p={p_lst_[i]}.csv"

            # with open(accu_txt_2_pth, 'w', encoding='UTF8', newline='') as f:
            #     writer = csv.writer(f)
            #     if K_or_T == 1:
            #         writer.writerow(["Coupling Strength"] + hori_arr.tolist())
            #     else:
            #         writer.writerow(["# of cycles"] + hori_arr.tolist())
            #     writer.writerow(["Adaptive LASSO"] + accu_arr_2_ada.tolist())
            #     writer.writerow(["LASSO"] + accu_arr_2_lasso.tolist())
            #     writer.writerow(["OLS"] + accu_arr_2_mle.tolist())
            #     f.close()
            #
            # with open(accu_txt_3_pth, 'w', encoding='UTF8', newline='') as f:
            #     writer = csv.writer(f)
            #     if K_or_T == 1:
            #         writer.writerow(["Coupling Strength"] + hori_arr.tolist())
            #     else:
            #         writer.writerow(["# of cycles"] + hori_arr.tolist())
            #     writer.writerow(["Adaptive LASSO"] + accu_arr_3_ada.tolist())
            #     writer.writerow(["LASSO"] + accu_arr_3_lasso.tolist())
            #     writer.writerow(["OLS"] + accu_arr_3_mle.tolist())
            #     f.close()
            #
            # with open(accu_txt_mix_pth, 'w', encoding='UTF8', newline='') as f:
            #     writer = csv.writer(f)
            #     if K_or_T == 1:
            #         writer.writerow(["Coupling Strength"] + hori_arr.tolist())
            #     else:
            #         writer.writerow(["# of cycles"] + hori_arr.tolist())
            #     writer.writerow(["Adaptive LASSO"] + accu_arr_mix_ada.tolist())
            #     writer.writerow(["LASSO"] + accu_arr_mix_lasso.tolist())
            #     writer.writerow(["OLS"] + accu_arr_mix_mle.tolist())
            #     f.close()

        # plt.rcParams["font.size"] = 36  # フォントを大きく
        plt.rcParams['font.family'] = 'Arial'
        params = {'legend.fontsize': 20,
                  'legend.handlelength': 1}
        plt.rcParams.update(params)
        fig.legend(ax, labels=["Adaptive LASSO", "LASSO", "OLS"], loc="lower right")
        fig.suptitle('Z-test of type accuracy', size=40)
        if K_or_T == 1:
            xlabel = 'Coupling strength'
        else:
            xlabel = 'Cycle number'
            ax[-1, 0].set_xscale("log")
            ax[-1, 1].set_xscale("log")
            ax[-1, 2].set_xscale("log")

        ax[-1, 0].set_xlabel(xlabel, fontsize=24)
        ax[-1, 0].set_xticks(hori_arr)
        ax[-1, 0].tick_params(axis='x', labelsize=18)
        ax[-1, 0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax[-1, 1].set_xlabel(xlabel, fontsize=24)
        ax[-1, 1].set_xticks(hori_arr)
        ax[-1, 1].tick_params(axis='x', labelsize=18)
        ax[-1, 1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax[-1, 2].set_xlabel(xlabel, fontsize=24)
        ax[-1, 2].set_xticks(hori_arr)
        ax[-1, 2].tick_params(axis='x', labelsize=18)
        ax[-1, 2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # ax.yaxis.set_label_coords(-0.08, 0.5)
        # ax.yaxis.set_major_formatter(formatter)

        # path = path_base + "MCC.jpg"
        # fig.savefig(path)


def create_task_type(conn_seed_, coup_seed_, noise_arr, natfreq_arr, K, T, pre_conn_2_arr=None, pre_conn_3_arr=None,
                     pre_reduce_arr=None, trial_num=5, dt=0.02, p_=0.1, noise_sth=0.2, ratio=3.0, st_fr=0.0, inf_l=1.0,
                     reduce_seed_=None, ops_limit=1.0, all_connected=False):
    task_lst = []
    two_lst = []
    three_lst = []
    mix_lst = []

    conn_reg = np.random.default_rng(conn_seed_)
    conn_arr2 = conn_reg.integers(0, 1e10, size=trial_num)
    if pre_conn_2_arr is not None:
        assert len(pre_conn_2_arr) == trial_num, "pre_conn_2_arr BOOMBA"
        for qw in range(len(pre_conn_2_arr)):
            if pre_conn_2_arr[qw] != 0:
                conn_arr2[qw] = pre_conn_2_arr[qw]
    conn_arr3 = conn_reg.integers(0, 1e10, size=trial_num)
    if pre_conn_3_arr is not None:
        assert len(pre_conn_3_arr) == trial_num, "pre_conn_3_arr BOOMBA"
        for qw in range(len(pre_conn_3_arr)):
            if pre_conn_3_arr[qw] != 0:
                conn_arr3[qw] = pre_conn_3_arr[qw]
    if coup_seed_ is not None:
        coup_arr = np.random.default_rng(coup_seed_).integers(0, 1e10, size=trial_num)
    else:
        coup_arr = np.full(trial_num, None, dtype=object)

    real_conn_2_seed = np.zeros(trial_num)
    real_conn_3_seed = np.zeros(trial_num)

    reduce_arr = np.zeros(trial_num)
    if pre_reduce_arr is not None:
        assert len(pre_reduce_arr) == trial_num
        for qw in range(len(pre_reduce_arr)):
            if pre_reduce_arr[qw] != 0:
                reduce_arr[qw] = pre_reduce_arr[qw]
    elif reduce_seed_ is not None:
        reduce_arr = np.random.default_rng(reduce_seed_).integers(0, 1e10, size=trial_num)
    else:
        raise Exception("Please specify reduce seed or array. ")

    test = 0
    for i in range(trial_num):
        two_model = GeneralInteraction(coupling2=K, coupling3=0, dt=dt, T=T, natfreqs=natfreq_arr, with_noise=True,
                                       noise_sth=noise_sth, normalize=True, conn2=p_, all_connected=all_connected,
                                       conn_seed=conn_arr2[i], coup_seed=coup_arr[i], noise_seed=noise_arr[0],
                                       starts_from=st_fr, inf_last=inf_l, old_legacy=True)
        # act_mat = two_model.run()
        # ax, _ = plot_phase_coherence(act_mat)
        # plt.show()

        while not check_larger_than_limit(two_model, ops_limit):
            new_seed = np.random.randint(0, 1e9)
            two_model = GeneralInteraction(coupling2=K, coupling3=0, dt=dt, T=T, natfreqs=natfreq_arr,
                                           with_noise=True,
                                           noise_sth=noise_sth, normalize=True, conn2=p_, all_connected=all_connected,
                                           conn_seed=new_seed, coup_seed=coup_arr[i], noise_seed=noise_arr[0],
                                           starts_from=st_fr, inf_last=inf_l, old_legacy=True)

        real_conn_2_seed[i] = two_model.conn_seed
        two_lst.append(two_model)
    task_lst.append(two_lst)
    conn_2_seed = merge_seed_array(conn_arr2, real_conn_2_seed)
    for i in range(trial_num):
        three_model = GeneralInteraction(coupling2=0, coupling3=K * ratio, dt=dt, T=T, natfreqs=natfreq_arr,
                                         with_noise=True, noise_sth=noise_sth, normalize=True, conn3=p_,
                                         all_connected=False, conn_seed=conn_arr3[i], coup_seed=coup_arr[i],
                                         noise_seed=noise_arr[1], starts_from=st_fr, inf_last=inf_l, old_legacy=True)

        while not check_larger_than_limit(three_model, ops_limit):
            new_seed = np.random.randint(0, 1e9)
            three_model = GeneralInteraction(coupling2=0, coupling3=K * ratio, dt=dt, T=T, natfreqs=natfreq_arr,
                                             with_noise=True, noise_sth=noise_sth, normalize=True, conn3=p_,
                                             all_connected=False, conn_seed=new_seed, coup_seed=coup_arr[i],
                                             noise_seed=noise_arr[1], starts_from=st_fr, inf_last=inf_l,
                                             old_legacy=True)

        real_conn_3_seed[i] = three_model.conn_seed
        three_lst.append(three_model)
    task_lst.append(three_lst)
    conn_3_seed = merge_seed_array(conn_arr3, real_conn_3_seed)

    low_lim = 0.8 * p_ / 2
    high_lim = 1.2 * p_ / 2
    for i in range(trial_num):
        conn_mat2 = task_lst[0][i].conn_mat2
        conn_mat3 = task_lst[1][i].conn_mat3
        reduced_2 = reduce_conn_2(conn_mat2, 0.5, reduce_seed=int(reduce_arr[i]))
        reduced_3 = reduce_conn_3(conn_mat3, 0.5, reduce_seed=int(reduce_arr[i]))
        reduced_2_coup = reduced_2 * (K / (p_ * len(natfreq_arr)))
        reduced_3_coup = reduced_3 * (K * ratio / (p_ * len(natfreq_arr) * len(natfreq_arr)))

        mix_model = GeneralInteraction(coupling2=K, coupling3=K * ratio, dt=dt, T=T, natfreqs=natfreq_arr,
                                       with_noise=True, noise_sth=noise_sth, conn=p_ / 2,
                                       pre_conn2=reduced_2, pre_conn3=reduced_3, pre_coup2=reduced_2_coup,
                                       pre_coup3=reduced_3_coup,
                                       noise_seed=noise_arr[2], starts_from=st_fr, inf_last=inf_l)

        conn2, _ = mix_model.get_connectivity()

        new_reduce_seed = reduce_arr[i]
        while not low_lim <= conn2 <= high_lim:
            conn_mat2 = task_lst[0][i].conn_mat2
            conn_mat3 = task_lst[1][i].conn_mat3
            new_reduce_seed = np.random.randint(0, 1e9)
            reduced_2 = reduce_conn_2(conn_mat2, 0.5, reduce_seed=new_reduce_seed)
            reduced_3 = reduce_conn_3(conn_mat3, 0.5, reduce_seed=new_reduce_seed)
            reduced_2_coup = reduced_2 * (K / (p_ * len(natfreq_arr)))
            reduced_3_coup = reduced_3 * (K * ratio / (p_ * len(natfreq_arr) * len(natfreq_arr)))

            mix_model = GeneralInteraction(dt=dt, T=T, natfreqs=natfreq_arr,
                                           with_noise=True, noise_sth=noise_sth, conn=p_ / 2,
                                           pre_conn2=reduced_2, pre_conn3=reduced_3, pre_coup2=reduced_2_coup,
                                           pre_coup3=reduced_3_coup,
                                           noise_seed=noise_arr[2], starts_from=st_fr, inf_last=inf_l)
            conn2, _ = mix_model.get_connectivity()
        reduce_arr[i] = new_reduce_seed
        mix_lst.append(mix_model)
    task_lst.append(mix_lst)
    return task_lst, conn_2_seed, conn_3_seed, reduce_arr


if __name__ == "__main__":
    natfreq_seed = 98764
    conn_seed = 20011105
    # conn_seed = None
    coup_seed = None
    noise_seed = 303943212
    # noise_seed = None
    reduce_seed = 2349231812

    start_datetime = datetime.now()
    p_lst = [0.10]

    my_data_2 = np.load("Typedata/new_two_seeds.npy")
    my_data_3 = np.load("Typedata/new_three_seeds.npy")
    my_data_reduced = np.load("Typedata/new_reduce_seeds.npy")
    main_for_type_test(natfreq_seed, conn_seed, coup_seed, noise_seed, reduce_seed=reduce_seed, K_or_T=1,
                       pre_conn_2_lst=my_data_2, pre_conn_3_lst=my_data_3, p_lst_=p_lst, trial_num=20,
                       pre_reduce_lst=my_data_reduced)
    # main_for_type_test(natfreq_seed, conn_seed, coup_seed, noise_seed, reduce_seed=reduce_seed, K_or_T=1,
    #                    p_lst_=p_lst, trial_num=1)
    # plt.show()

    now = datetime.now()
    duration = now - start_datetime
    print("Duration is =", duration)
