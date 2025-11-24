from Fig_3 import check_larger_than_limit, merge_seed_array
from GeneralInteraction import *


def shrink_two_arr(arr1, arr2):
    assert len(arr1) == len(arr2), "Two arrays need to have same lengths. "

    arr1_output = np.copy(arr1)
    arr2_output = np.copy(arr2)

    need_delete = np.array([])

    j = 0
    for i in range(len(arr1)):
        if arr1[i] == 0 and arr2[i] == 0:
            j = 1
            need_delete = np.append(need_delete, i)

    while len(need_delete) > 0:
        arr1_output = np.delete(arr1_output, int(need_delete[0]))
        arr2_output = np.delete(arr2_output, int(need_delete[0]))
        need_delete = need_delete - 1
        need_delete = np.delete(need_delete, 0)

    assert len(arr1_output) == len(arr2_output), "Boom!!!"

    if j == 1:
        arr1_output = np.append(arr1_output, 0)
        arr2_output = np.append(arr2_output, 0)

    return arr1_output, arr2_output


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes_):
        low_x, high_x = axes_.get_xlim()
        low_y, high_y = axes_.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def make_dirs(p_):
    now_ = datetime.now()
    date = now_.strftime("%Y%m%d")
    tm_string = now_.strftime("%H%M%S")
    file_path1 = "Fig4data/img/" + date + "/"
    os.makedirs(file_path1, exist_ok=True)

    file_path_p = "Fig4data/img/" + date + "/" + tm_string + f"_p={str(p_)}"
    os.mkdir(file_path_p)
    return file_path_p


def create_task_Fig4(conn_seed_, coup_seed_, noise_arr, natfreq_arr, K, T, pre_conn_2_arr=None, pre_conn_3_arr=None,
                     trial_num=5, dt=0.02, p_=0.1, noise_sth=0.2, ratio=6.0, st_fr=0.0, inf_l=1.0, reduce_seed=None,
                     ops_limit=1.0, all_connected=False):
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

    for i in range(trial_num):
        two_model = GeneralInteraction(coupling2=K, coupling3=0, dt=dt, T=T, natfreqs=natfreq_arr, with_noise=True,
                                       noise_sth=noise_sth, normalize=True, conn2=p_, all_connected=all_connected,
                                       conn_seed=conn_arr2[i], coup_seed=coup_arr[i], noise_seed=noise_arr[0],
                                       starts_from=st_fr, inf_last=inf_l, random_coup2=True, old_legacy=True)

        while not check_larger_than_limit(two_model, ops_limit):
            new_seed = np.random.randint(0, 1e10)
            two_model = GeneralInteraction(coupling2=K, coupling3=0, dt=dt, T=T, natfreqs=natfreq_arr,
                                           with_noise=True,
                                           noise_sth=noise_sth, normalize=True, conn2=p_, all_connected=all_connected,
                                           conn_seed=new_seed, coup_seed=coup_arr[i], noise_seed=noise_arr[0],
                                           starts_from=st_fr, inf_last=inf_l, random_coup2=True)

        if conn_arr2[i] != two_model.conn_seed:
            conn_arr2[i] = two_model.conn_seed
            real_conn_2_seed[i] = two_model.conn_seed
        two_lst.append(two_model)
    task_lst.append(two_lst)
    conn_2_seed = merge_seed_array(conn_arr2, real_conn_2_seed)
    for i in range(trial_num):
        three_model = GeneralInteraction(coupling2=0, coupling3=K * ratio, dt=dt, T=T, natfreqs=natfreq_arr,
                                         with_noise=True, noise_sth=noise_sth, normalize=True, conn3=p_,
                                         all_connected=False, conn_seed=conn_arr3[i], coup_seed=coup_arr[i],
                                         noise_seed=noise_arr[1], starts_from=st_fr, inf_last=inf_l, random_coup3=True,
                                         old_legacy=True)

        while not check_larger_than_limit(three_model, ops_limit):
            new_seed = np.random.randint(0, 1e10)
            three_model = GeneralInteraction(coupling2=0, coupling3=K * ratio, dt=dt, T=T, natfreqs=natfreq_arr,
                                             with_noise=True, noise_sth=noise_sth, normalize=True, conn3=p_,
                                             all_connected=False, conn_seed=new_seed, coup_seed=coup_arr[i],
                                             noise_seed=noise_arr[1], starts_from=st_fr, inf_last=inf_l,
                                             random_coup3=True, old_legacy=True)

        if conn_arr3[i] != three_model.conn_seed:
            conn_arr3[i] = three_model.conn_seed
            real_conn_3_seed[i] = three_model.conn_seed
        three_lst.append(three_model)
    task_lst.append(three_lst)
    conn_3_seed = merge_seed_array(conn_arr3, real_conn_3_seed)
    for i in range(trial_num):
        conn_mat2 = task_lst[0][i].conn_mat2
        conn_mat3 = task_lst[1][i].conn_mat3
        coup_mat2 = task_lst[0][i].coupling2
        coup_mat3 = task_lst[1][i].coupling3
        reduced_2 = reduce_conn_2(conn_mat2, 0.5, reduce_seed=reduce_seed)
        reduced_3 = reduce_conn_3(conn_mat3, 0.5, reduce_seed=reduce_seed)
        reduced_2_coup = reduced_2 * coup_mat2
        reduced_3_coup = reduced_3 * coup_mat3

        mix_model = GeneralInteraction(dt=dt, T=T, natfreqs=natfreq_arr,
                                       with_noise=True, noise_sth=noise_sth, conn=p_ / 2,
                                       pre_conn2=reduced_2, pre_conn3=reduced_3, pre_coup2=reduced_2_coup,
                                       pre_coup3=reduced_3_coup,
                                       noise_seed=noise_arr[2], starts_from=st_fr, inf_last=inf_l,
                                       random_coup2=True, random_coup3=True)

        mix_lst.append(mix_model)
    task_lst.append(mix_lst)
    return task_lst, conn_2_seed, conn_3_seed


def main_for_Fig4(natfreq_seed_, conn_seed_, coup_seed_, noise_seed_, pre_conn_2_arr=None, pre_conn_3_arr=None, p_=0.1,
                  r2=False, mcc=False, weak_strong=0.0):
    # 1. uni parameters
    trial_num = 10
    n_nodes = 12
    remove_tail_r2 = 0
    tp_only = False

    dt = 0.02
    noise_sth = 0.2
    st_fr = 1 / 9
    inf_l = 2 / 3
    # filter variable
    ops_limit = 0.4
    reduce_seed = 19374759102
    ratio = 6.0
    all_connected = False
    T = 900
    K = 0.1

    # _, paths = make_dirs(p_, K_or_T)

    # 2. variable (K or T) (1 is K, 2 is T)
    natfreq_arr = np.random.default_rng(natfreq_seed_).normal(loc=1, scale=0.1, size=n_nodes)
    noise_arr = np.random.default_rng(noise_seed_).integers(0, 1e10, size=3)

    model_lst, real_conn_2_arr, real_conn_3_arr = create_task_Fig4(conn_seed_=conn_seed_, coup_seed_=coup_seed_,
                                                                   noise_arr=noise_arr, natfreq_arr=natfreq_arr,
                                                                   pre_conn_2_arr=pre_conn_2_arr,
                                                                   trial_num=trial_num, K=K, T=T, dt=dt, p_=p_,
                                                                   noise_sth=noise_sth, ratio=ratio, st_fr=st_fr,
                                                                   inf_l=inf_l, reduce_seed=reduce_seed,
                                                                   ops_limit=ops_limit, all_connected=all_connected,
                                                                   pre_conn_3_arr=pre_conn_3_arr)
    if real_conn_2_arr.any():
        print("The outcome conn2_arr is", real_conn_2_arr)
    real_2_lst = real_conn_2_arr.tolist()
    if real_conn_3_arr.any():
        print("The outcome conn3_arr is", real_conn_3_arr)
    real_3_lst = real_conn_3_arr.tolist()

    assert len(model_lst) == 3, "len of model_lst has to be 3, pairwise, 3-int, and mixture. "

    demo_2_0 = np.array([])
    mle_2_0 = np.array([])
    lasso_2_0 = np.array([])
    ada_2_0 = np.array([])
    demo_3_0 = np.array([])
    mle_3_0 = np.array([])
    lasso_3_0 = np.array([])
    ada_3_0 = np.array([])

    demo_2_1 = np.array([])
    mle_2_1 = np.array([])
    lasso_2_1 = np.array([])
    ada_2_1 = np.array([])
    demo_3_1 = np.array([])
    mle_3_1 = np.array([])
    lasso_3_1 = np.array([])
    ada_3_1 = np.array([])

    demo_2_2 = np.array([])
    mle_2_2 = np.array([])
    lasso_2_2 = np.array([])
    ada_2_2 = np.array([])
    demo_3_2 = np.array([])
    mle_3_2 = np.array([])
    lasso_3_2 = np.array([])
    ada_3_2 = np.array([])

    if r2:
        mse_2_mle_arr = np.array([])
        mse_2_lasso_arr = np.array([])
        mse_2_ada_arr = np.array([])
        mse_3_mle_arr = np.array([])
        mse_3_lasso_arr = np.array([])
        mse_3_ada_arr = np.array([])
        mse_mix_mle_arr = np.array([])
        mse_mix_lasso_arr = np.array([])
        mse_mix_ada_arr = np.array([])

        R2_2_mle_arr = np.array([])
        R2_2_lasso_arr = np.array([])
        R2_2_ada_arr = np.array([])
        R2_3_mle_arr = np.array([])
        R2_3_lasso_arr = np.array([])
        R2_3_ada_arr = np.array([])
        R2_mix_mle_arr = np.array([])
        R2_mix_lasso_arr = np.array([])
        R2_mix_ada_arr = np.array([])

        pcc_2_mle_arr = np.array([])
        pcc_2_lasso_arr = np.array([])
        pcc_2_ada_arr = np.array([])
        pcc_3_mle_arr = np.array([])
        pcc_3_lasso_arr = np.array([])
        pcc_3_ada_arr = np.array([])
        pcc_mix_mle_arr = np.array([])
        pcc_mix_lasso_arr = np.array([])
        pcc_mix_ada_arr = np.array([])

    if mcc:
        mcc_rates_2_mle = np.zeros(trial_num)
        mcc_rates_2_lasso = np.zeros(trial_num)
        mcc_rates_2_ada = np.zeros(trial_num)
        mcc_rates_3_mle = np.zeros(trial_num)
        mcc_rates_3_lasso = np.zeros(trial_num)
        mcc_rates_3_ada = np.zeros(trial_num)
        mcc_rates_mix_mle = np.zeros(trial_num)
        mcc_rates_mix_lasso = np.zeros(trial_num)
        mcc_rates_mix_ada = np.zeros(trial_num)

        mcc_rates_2_mle_2 = np.zeros(trial_num)
        mcc_rates_2_lasso_2 = np.zeros(trial_num)
        mcc_rates_2_ada_2 = np.zeros(trial_num)
        mcc_rates_3_mle_2 = np.zeros(trial_num)
        mcc_rates_3_lasso_2 = np.zeros(trial_num)
        mcc_rates_3_ada_2 = np.zeros(trial_num)
        mcc_rates_mix_mle_2 = np.zeros(trial_num)
        mcc_rates_mix_lasso_2 = np.zeros(trial_num)
        mcc_rates_mix_ada_2 = np.zeros(trial_num)

        mcc_rates_2_mle_3 = np.zeros(trial_num)
        mcc_rates_2_lasso_3 = np.zeros(trial_num)
        mcc_rates_2_ada_3 = np.zeros(trial_num)
        mcc_rates_3_mle_3 = np.zeros(trial_num)
        mcc_rates_3_lasso_3 = np.zeros(trial_num)
        mcc_rates_3_ada_3 = np.zeros(trial_num)
        mcc_rates_mix_mle_3 = np.zeros(trial_num)
        mcc_rates_mix_lasso_3 = np.zeros(trial_num)
        mcc_rates_mix_ada_3 = np.zeros(trial_num)

        if weak_strong > 0:
            mcc_rates_2_mle_2_weak = np.zeros(trial_num)
            mcc_rates_2_lasso_2_weak = np.zeros(trial_num)
            mcc_rates_2_ada_2_weak = np.zeros(trial_num)
            mcc_rates_3_mle_2_weak = np.zeros(trial_num)
            mcc_rates_3_lasso_2_weak = np.zeros(trial_num)
            mcc_rates_3_ada_2_weak = np.zeros(trial_num)
            mcc_rates_mix_mle_2_weak = np.zeros(trial_num)
            mcc_rates_mix_lasso_2_weak = np.zeros(trial_num)
            mcc_rates_mix_ada_2_weak = np.zeros(trial_num)

            mcc_rates_2_mle_3_weak = np.zeros(trial_num)
            mcc_rates_2_lasso_3_weak = np.zeros(trial_num)
            mcc_rates_2_ada_3_weak = np.zeros(trial_num)
            mcc_rates_3_mle_3_weak = np.zeros(trial_num)
            mcc_rates_3_lasso_3_weak = np.zeros(trial_num)
            mcc_rates_3_ada_3_weak = np.zeros(trial_num)
            mcc_rates_mix_mle_3_weak = np.zeros(trial_num)
            mcc_rates_mix_lasso_3_weak = np.zeros(trial_num)
            mcc_rates_mix_ada_3_weak = np.zeros(trial_num)

            mcc_rates_2_mle_weak = np.zeros(trial_num)
            mcc_rates_2_lasso_weak = np.zeros(trial_num)
            mcc_rates_2_ada_weak = np.zeros(trial_num)
            mcc_rates_3_mle_weak = np.zeros(trial_num)
            mcc_rates_3_lasso_weak = np.zeros(trial_num)
            mcc_rates_3_ada_weak = np.zeros(trial_num)
            mcc_rates_mix_mle_weak = np.zeros(trial_num)
            mcc_rates_mix_lasso_weak = np.zeros(trial_num)
            mcc_rates_mix_ada_weak = np.zeros(trial_num)

            mcc_rates_2_mle_2_strong = np.zeros(trial_num)
            mcc_rates_2_lasso_2_strong = np.zeros(trial_num)
            mcc_rates_2_ada_2_strong = np.zeros(trial_num)
            mcc_rates_3_mle_2_strong = np.zeros(trial_num)
            mcc_rates_3_lasso_2_strong = np.zeros(trial_num)
            mcc_rates_3_ada_2_strong = np.zeros(trial_num)
            mcc_rates_mix_mle_2_strong = np.zeros(trial_num)
            mcc_rates_mix_lasso_2_strong = np.zeros(trial_num)
            mcc_rates_mix_ada_2_strong = np.zeros(trial_num)

            mcc_rates_2_mle_3_strong = np.zeros(trial_num)
            mcc_rates_2_lasso_3_strong = np.zeros(trial_num)
            mcc_rates_2_ada_3_strong = np.zeros(trial_num)
            mcc_rates_3_mle_3_strong = np.zeros(trial_num)
            mcc_rates_3_lasso_3_strong = np.zeros(trial_num)
            mcc_rates_3_ada_3_strong = np.zeros(trial_num)
            mcc_rates_mix_mle_3_strong = np.zeros(trial_num)
            mcc_rates_mix_lasso_3_strong = np.zeros(trial_num)
            mcc_rates_mix_ada_3_strong = np.zeros(trial_num)

            mcc_rates_2_mle_strong = np.zeros(trial_num)
            mcc_rates_2_lasso_strong = np.zeros(trial_num)
            mcc_rates_2_ada_strong = np.zeros(trial_num)
            mcc_rates_3_mle_strong = np.zeros(trial_num)
            mcc_rates_3_lasso_strong = np.zeros(trial_num)
            mcc_rates_3_ada_strong = np.zeros(trial_num)
            mcc_rates_mix_mle_strong = np.zeros(trial_num)
            mcc_rates_mix_lasso_strong = np.zeros(trial_num)
            mcc_rates_mix_ada_strong = np.zeros(trial_num)

    for j in range(trial_num):
        act_mat = model_lst[0][j].run()
        prepare_all = model_lst[0][j].prepare_diffs(act_mat)
        mle_or_ols_results = model_lst[0][j].solve_ols(all_prepared=prepare_all)
        ada_results = model_lst[0][j].solve_ada_lasso(all_prepared=prepare_all, mle_or_ols_results=mle_or_ols_results)
        lasso_results = model_lst[0][j].solve_lasso(all_prepared=prepare_all)
        all_results, conn_all = model_lst[0][j].conn_criteria_base(mle_or_ols_results=mle_or_ols_results,
                                                                   lasso_results=lasso_results,
                                                                   ada_results=ada_results, conn_out=True,
                                                                   weak_strong=weak_strong)
        # mle_or_ols_results = model_lst[0][j].mle_with_fdr(mle_or_ols_results, conn_all)
        demo_2, demo_3, mle_2, mle_3, lasso_2, lasso_3, ada_2, ada_3 = \
            model_lst[0][j].coup_compare_base(mle_or_ols_results=mle_or_ols_results, lasso_results=lasso_results,
                                              ada_results=ada_results)

        demo_2_0 = np.concatenate((demo_2_0, demo_2), axis=0)
        mle_2_0 = np.concatenate((mle_2_0, mle_2), axis=0)
        lasso_2_0 = np.concatenate((lasso_2_0, lasso_2), axis=0)
        ada_2_0 = np.concatenate((ada_2_0, ada_2), axis=0)
        demo_3_0 = np.concatenate((demo_3_0, demo_3), axis=0)
        mle_3_0 = np.concatenate((mle_3_0, mle_3), axis=0)
        lasso_3_0 = np.concatenate((lasso_3_0, lasso_3), axis=0)
        ada_3_0 = np.concatenate((ada_3_0, ada_3), axis=0)

        if all_results["mle"] is not None and mcc:
            mcc_rates_2_mle[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP"], all_results["mle"]["TN"],
                                                          all_results["mle"]["FP"], all_results["mle"]["FN"])
            mcc_rates_2_mle_2[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP_2"], all_results["mle"]["TN_2"],
                                                            all_results["mle"]["FP_2"], all_results["mle"]["FN_2"])
            mcc_rates_2_mle_3[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP_3"], all_results["mle"]["TN_3"],
                                                            all_results["mle"]["FP_3"], all_results["mle"]["FN_3"])
            if weak_strong > 0:
                mcc_rates_2_mle_2_weak[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP_2_weak"],
                                                                     all_results["mle"]["TN_2_weak"],
                                                                     all_results["mle"]["FP_2_weak"],
                                                                     all_results["mle"]["FN_2_weak"])
                mcc_rates_2_mle_3_weak[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP_3_weak"],
                                                                     all_results["mle"]["TN_3_weak"],
                                                                     all_results["mle"]["FP_3_weak"],
                                                                     all_results["mle"]["FN_3_weak"])
                mcc_rates_2_mle_weak[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP_weak"],
                                                                   all_results["mle"]["TN_weak"],
                                                                   all_results["mle"]["FP_weak"],
                                                                   all_results["mle"]["FN_weak"])
                mcc_rates_2_mle_2_strong[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP_2_strong"],
                                                                       all_results["mle"]["TN_2_strong"],
                                                                       all_results["mle"]["FP_2_strong"],
                                                                       all_results["mle"]["FN_2_strong"])
                mcc_rates_2_mle_3_strong[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP_3_strong"],
                                                                       all_results["mle"]["TN_3_strong"],
                                                                       all_results["mle"]["FP_3_strong"],
                                                                       all_results["mle"]["FN_3_strong"])
                mcc_rates_2_mle_strong[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP_strong"],
                                                                     all_results["mle"]["TN_strong"],
                                                                     all_results["mle"]["FP_strong"],
                                                                     all_results["mle"]["FN_strong"])

        if all_results["lasso"] is not None and mcc:
            mcc_rates_2_lasso[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP"], all_results["lasso"]["TN"],
                                                            all_results["lasso"]["FP"], all_results["lasso"]["FN"])
            mcc_rates_2_lasso_2[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP_2"],
                                                              all_results["lasso"]["TN_2"],
                                                              all_results["lasso"]["FP_2"],
                                                              all_results["lasso"]["FN_2"])
            mcc_rates_2_lasso_3[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP_3"],
                                                              all_results["lasso"]["TN_3"],
                                                              all_results["lasso"]["FP_3"],
                                                              all_results["lasso"]["FN_3"])

            if weak_strong > 0:
                mcc_rates_2_lasso_2_weak[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP_2_weak"],
                                                                       all_results["lasso"]["TN_2_weak"],
                                                                       all_results["lasso"]["FP_2_weak"],
                                                                       all_results["lasso"]["FN_2_weak"])
                mcc_rates_2_lasso_3_weak[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP_3_weak"],
                                                                       all_results["lasso"]["TN_3_weak"],
                                                                       all_results["lasso"]["FP_3_weak"],
                                                                       all_results["lasso"]["FN_3_weak"])
                mcc_rates_2_lasso_weak[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP_weak"],
                                                                     all_results["lasso"]["TN_weak"],
                                                                     all_results["lasso"]["FP_weak"],
                                                                     all_results["lasso"]["FN_weak"])
                mcc_rates_2_lasso_2_strong[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP_2_strong"],
                                                                         all_results["lasso"]["TN_2_strong"],
                                                                         all_results["lasso"]["FP_2_strong"],
                                                                         all_results["lasso"]["FN_2_strong"])
                mcc_rates_2_lasso_3_strong[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP_3_strong"],
                                                                         all_results["lasso"]["TN_3_strong"],
                                                                         all_results["lasso"]["FP_3_strong"],
                                                                         all_results["lasso"]["FN_3_strong"])
                mcc_rates_2_lasso_strong[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP_strong"],
                                                                       all_results["lasso"]["TN_strong"],
                                                                       all_results["lasso"]["FP_strong"],
                                                                       all_results["lasso"]["FN_strong"])

        if all_results["ada"] is not None and mcc:
            mcc_rates_2_ada[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP"], all_results["ada"]["TN"],
                                                          all_results["ada"]["FP"], all_results["ada"]["FN"])
            mcc_rates_2_ada_2[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP_2"], all_results["ada"]["TN_2"],
                                                            all_results["ada"]["FP_2"], all_results["ada"]["FN_2"])
            mcc_rates_2_ada_3[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP_3"], all_results["ada"]["TN_3"],
                                                            all_results["ada"]["FP_3"], all_results["ada"]["FN_3"])

            if weak_strong > 0:
                mcc_rates_2_ada_2_weak[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP_2_weak"],
                                                                     all_results["ada"]["TN_2_weak"],
                                                                     all_results["ada"]["FP_2_weak"],
                                                                     all_results["ada"]["FN_2_weak"])
                mcc_rates_2_ada_3_weak[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP_3_weak"],
                                                                     all_results["ada"]["TN_3_weak"],
                                                                     all_results["ada"]["FP_3_weak"],
                                                                     all_results["ada"]["FN_3_weak"])
                mcc_rates_2_ada_weak[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP_weak"],
                                                                   all_results["ada"]["TN_weak"],
                                                                   all_results["ada"]["FP_weak"],
                                                                   all_results["ada"]["FN_weak"])
                mcc_rates_2_ada_2_strong[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP_2_strong"],
                                                                       all_results["ada"]["TN_2_strong"],
                                                                       all_results["ada"]["FP_2_strong"],
                                                                       all_results["ada"]["FN_2_strong"])
                mcc_rates_2_ada_3_strong[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP_3_strong"],
                                                                       all_results["ada"]["TN_3_strong"],
                                                                       all_results["ada"]["FP_3_strong"],
                                                                       all_results["ada"]["FN_3_strong"])
                mcc_rates_2_ada_strong[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP_strong"],
                                                                     all_results["ada"]["TN_strong"],
                                                                     all_results["ada"]["FP_strong"],
                                                                     all_results["ada"]["FN_strong"])

        if r2:
            mse_2_mle, r2_2_mle, pcc_2_mle = model_lst[0][j].mse_and_r2_combined(mle_or_ols_results, remove_tail=remove_tail_r2,
                                                                         tp_only=tp_only)
            mse_2_lasso, r2_2_lasso, pcc_2_lasso = model_lst[0][j].mse_and_r2_combined(lasso_results, remove_tail=remove_tail_r2,
                                                                             tp_only=tp_only)
            mse_2_ada, r2_2_ada, pcc_2_ada = model_lst[0][j].mse_and_r2_combined(ada_results, remove_tail=remove_tail_r2,
                                                                         tp_only=tp_only)
            mse_2_mle_arr = np.append(mse_2_mle_arr, mse_2_mle)
            mse_2_lasso_arr = np.append(mse_2_lasso_arr, mse_2_lasso)
            mse_2_ada_arr = np.append(mse_2_ada_arr, mse_2_ada)
            R2_2_mle_arr = np.append(R2_2_mle_arr, r2_2_mle)
            R2_2_lasso_arr = np.append(R2_2_lasso_arr, r2_2_lasso)
            R2_2_ada_arr = np.append(R2_2_ada_arr, r2_2_ada)
            pcc_2_mle_arr = np.append(pcc_2_mle_arr, pcc_2_mle)
            pcc_2_lasso_arr = np.append(pcc_2_lasso_arr, pcc_2_lasso)
            pcc_2_ada_arr = np.append(pcc_2_ada_arr, pcc_2_ada)

        act_mat = model_lst[1][j].run()
        prepare_all = model_lst[1][j].prepare_diffs(act_mat)
        mle_or_ols_results = model_lst[1][j].solve_ols(all_prepared=prepare_all)
        ada_results = model_lst[1][j].solve_ada_lasso(all_prepared=prepare_all, mle_or_ols_results=mle_or_ols_results)
        lasso_results = model_lst[1][j].solve_lasso(all_prepared=prepare_all)
        all_results, conn_all = model_lst[1][j].conn_criteria_base(mle_or_ols_results=mle_or_ols_results,
                                                                   lasso_results=lasso_results,
                                                                   ada_results=ada_results, conn_out=True,
                                                                   weak_strong=weak_strong)
        # mle_or_ols_results = model_lst[1][j].mle_with_fdr(mle_or_ols_results, conn_all)
        demo_2, demo_3, mle_2, mle_3, lasso_2, lasso_3, ada_2, ada_3 = \
            model_lst[1][j].coup_compare_base(mle_or_ols_results=mle_or_ols_results, lasso_results=lasso_results,
                                              ada_results=ada_results)

        demo_2_1 = np.concatenate((demo_2_1, demo_2), axis=0)
        mle_2_1 = np.concatenate((mle_2_1, mle_2), axis=0)
        lasso_2_1 = np.concatenate((lasso_2_1, lasso_2), axis=0)
        ada_2_1 = np.concatenate((ada_2_1, ada_2), axis=0)
        demo_3_1 = np.concatenate((demo_3_1, demo_3), axis=0)
        mle_3_1 = np.concatenate((mle_3_1, mle_3), axis=0)
        lasso_3_1 = np.concatenate((lasso_3_1, lasso_3), axis=0)
        ada_3_1 = np.concatenate((ada_3_1, ada_3), axis=0)

        if all_results["mle"] is not None and mcc:
            mcc_rates_3_mle[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP"], all_results["mle"]["TN"],
                                                          all_results["mle"]["FP"], all_results["mle"]["FN"])
            mcc_rates_3_mle_2[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP_2"], all_results["mle"]["TN_2"],
                                                            all_results["mle"]["FP_2"], all_results["mle"]["FN_2"])
            mcc_rates_3_mle_3[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP_3"], all_results["mle"]["TN_3"],
                                                            all_results["mle"]["FP_3"], all_results["mle"]["FN_3"])

            if weak_strong > 0:
                mcc_rates_3_mle_2_weak[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP_2_weak"],
                                                                     all_results["mle"]["TN_2_weak"],
                                                                     all_results["mle"]["FP_2_weak"],
                                                                     all_results["mle"]["FN_2_weak"])
                mcc_rates_3_mle_3_weak[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP_3_weak"],
                                                                     all_results["mle"]["TN_3_weak"],
                                                                     all_results["mle"]["FP_3_weak"],
                                                                     all_results["mle"]["FN_3_weak"])
                mcc_rates_3_mle_weak[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP_weak"],
                                                                   all_results["mle"]["TN_weak"],
                                                                   all_results["mle"]["FP_weak"],
                                                                   all_results["mle"]["FN_weak"])
                mcc_rates_3_mle_2_strong[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP_2_strong"],
                                                                       all_results["mle"]["TN_2_strong"],
                                                                       all_results["mle"]["FP_2_strong"],
                                                                       all_results["mle"]["FN_2_strong"])
                mcc_rates_3_mle_3_strong[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP_3_strong"],
                                                                       all_results["mle"]["TN_3_strong"],
                                                                       all_results["mle"]["FP_3_strong"],
                                                                       all_results["mle"]["FN_3_strong"])
                mcc_rates_3_mle_strong[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP_strong"],
                                                                     all_results["mle"]["TN_strong"],
                                                                     all_results["mle"]["FP_strong"],
                                                                     all_results["mle"]["FN_strong"])
        if all_results["lasso"] is not None and mcc:
            mcc_rates_3_lasso[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP"], all_results["lasso"]["TN"],
                                                            all_results["lasso"]["FP"], all_results["lasso"]["FN"])
            mcc_rates_3_lasso_2[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP_2"],
                                                              all_results["lasso"]["TN_2"],
                                                              all_results["lasso"]["FP_2"],
                                                              all_results["lasso"]["FN_2"])
            mcc_rates_3_lasso_3[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP_3"],
                                                              all_results["lasso"]["TN_3"],
                                                              all_results["lasso"]["FP_3"],
                                                              all_results["lasso"]["FN_3"])

            if weak_strong > 0:
                mcc_rates_3_lasso_2_weak[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP_2_weak"],
                                                                       all_results["lasso"]["TN_2_weak"],
                                                                       all_results["lasso"]["FP_2_weak"],
                                                                       all_results["lasso"]["FN_2_weak"])
                mcc_rates_3_lasso_3_weak[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP_3_weak"],
                                                                       all_results["lasso"]["TN_3_weak"],
                                                                       all_results["lasso"]["FP_3_weak"],
                                                                       all_results["lasso"]["FN_3_weak"])
                mcc_rates_3_lasso_weak[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP_weak"],
                                                                     all_results["lasso"]["TN_weak"],
                                                                     all_results["lasso"]["FP_weak"],
                                                                     all_results["lasso"]["FN_weak"])
                mcc_rates_3_lasso_2_strong[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP_2_strong"],
                                                                         all_results["lasso"]["TN_2_strong"],
                                                                         all_results["lasso"]["FP_2_strong"],
                                                                         all_results["lasso"]["FN_2_strong"])
                mcc_rates_3_lasso_3_strong[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP_3_strong"],
                                                                         all_results["lasso"]["TN_3_strong"],
                                                                         all_results["lasso"]["FP_3_strong"],
                                                                         all_results["lasso"]["FN_3_strong"])
                mcc_rates_3_lasso_strong[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP_strong"],
                                                                       all_results["lasso"]["TN_strong"],
                                                                       all_results["lasso"]["FP_strong"],
                                                                       all_results["lasso"]["FN_strong"])
        if all_results["ada"] is not None and mcc:
            mcc_rates_3_ada[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP"], all_results["ada"]["TN"],
                                                          all_results["ada"]["FP"], all_results["ada"]["FN"])
            mcc_rates_3_ada_2[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP_2"], all_results["ada"]["TN_2"],
                                                            all_results["ada"]["FP_2"], all_results["ada"]["FN_2"])
            mcc_rates_3_ada_3[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP_3"], all_results["ada"]["TN_3"],
                                                            all_results["ada"]["FP_3"], all_results["ada"]["FN_3"])

            if weak_strong > 0:
                mcc_rates_3_ada_2_weak[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP_2_weak"],
                                                                     all_results["ada"]["TN_2_weak"],
                                                                     all_results["ada"]["FP_2_weak"],
                                                                     all_results["ada"]["FN_2_weak"])
                mcc_rates_3_ada_3_weak[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP_3_weak"],
                                                                     all_results["ada"]["TN_3_weak"],
                                                                     all_results["ada"]["FP_3_weak"],
                                                                     all_results["ada"]["FN_3_weak"])
                mcc_rates_3_ada_weak[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP_weak"],
                                                                   all_results["ada"]["TN_weak"],
                                                                   all_results["ada"]["FP_weak"],
                                                                   all_results["ada"]["FN_weak"])
                mcc_rates_3_ada_2_strong[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP_2_strong"],
                                                                       all_results["ada"]["TN_2_strong"],
                                                                       all_results["ada"]["FP_2_strong"],
                                                                       all_results["ada"]["FN_2_strong"])
                mcc_rates_3_ada_3_strong[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP_3_strong"],
                                                                       all_results["ada"]["TN_3_strong"],
                                                                       all_results["ada"]["FP_3_strong"],
                                                                       all_results["ada"]["FN_3_strong"])
                mcc_rates_3_ada_strong[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP_strong"],
                                                                     all_results["ada"]["TN_strong"],
                                                                     all_results["ada"]["FP_strong"],
                                                                     all_results["ada"]["FN_strong"])

        if r2:
            mse_3_mle, r2_3_mle, pcc_3_mle = model_lst[1][j].mse_and_r2_combined(mle_or_ols_results, remove_tail=remove_tail_r2,
                                                                         tp_only=tp_only)
            mse_3_lasso, r2_3_lasso, pcc_3_lasso = model_lst[1][j].mse_and_r2_combined(lasso_results, remove_tail=remove_tail_r2,
                                                                             tp_only=tp_only)
            mse_3_ada, r2_3_ada, pcc_3_ada = model_lst[1][j].mse_and_r2_combined(ada_results, remove_tail=remove_tail_r2,
                                                                         tp_only=tp_only)
            mse_3_mle_arr = np.append(mse_3_mle_arr, mse_3_mle)
            mse_3_lasso_arr = np.append(mse_3_lasso_arr, mse_3_lasso)
            mse_3_ada_arr = np.append(mse_3_ada_arr, mse_3_ada)
            R2_3_mle_arr = np.append(R2_3_mle_arr, r2_3_mle)
            R2_3_lasso_arr = np.append(R2_3_lasso_arr, r2_3_lasso)
            R2_3_ada_arr = np.append(R2_3_ada_arr, r2_3_ada)
            pcc_3_mle_arr = np.append(pcc_3_mle_arr, pcc_3_mle)
            pcc_3_lasso_arr = np.append(pcc_3_lasso_arr, pcc_3_lasso)
            pcc_3_ada_arr = np.append(pcc_3_ada_arr, pcc_3_ada)

        act_mat = model_lst[2][j].run()
        prepare_all = model_lst[2][j].prepare_diffs(act_mat)
        mle_or_ols_results = model_lst[2][j].solve_ols(all_prepared=prepare_all)
        ada_results = model_lst[2][j].solve_ada_lasso(all_prepared=prepare_all, mle_or_ols_results=mle_or_ols_results)
        lasso_results = model_lst[2][j].solve_lasso(all_prepared=prepare_all)
        all_results, conn_all = model_lst[2][j].conn_criteria_base(mle_or_ols_results=mle_or_ols_results,
                                                                   lasso_results=lasso_results,
                                                                   ada_results=ada_results, conn_out=True,
                                                                   weak_strong=weak_strong)
        # mle_or_ols_results = model_lst[2][j].mle_with_fdr(mle_or_ols_results, conn_all)
        demo_2, demo_3, mle_2, mle_3, lasso_2, lasso_3, ada_2, ada_3 = \
            model_lst[2][j].coup_compare_base(mle_or_ols_results=mle_or_ols_results, lasso_results=lasso_results,
                                              ada_results=ada_results)

        demo_2_2 = np.concatenate((demo_2_2, demo_2), axis=0)
        mle_2_2 = np.concatenate((mle_2_2, mle_2), axis=0)
        lasso_2_2 = np.concatenate((lasso_2_2, lasso_2), axis=0)
        ada_2_2 = np.concatenate((ada_2_2, ada_2), axis=0)
        demo_3_2 = np.concatenate((demo_3_2, demo_3), axis=0)
        mle_3_2 = np.concatenate((mle_3_2, mle_3), axis=0)
        lasso_3_2 = np.concatenate((lasso_3_2, lasso_3), axis=0)
        ada_3_2 = np.concatenate((ada_3_2, ada_3), axis=0)

        if all_results["mle"] is not None and mcc:
            mcc_rates_mix_mle[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP"], all_results["mle"]["TN"],
                                                            all_results["mle"]["FP"], all_results["mle"]["FN"])
            mcc_rates_mix_mle_2[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP_2"], all_results["mle"]["TN_2"],
                                                              all_results["mle"]["FP_2"], all_results["mle"]["FN_2"])
            mcc_rates_mix_mle_3[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP_3"], all_results["mle"]["TN_3"],
                                                              all_results["mle"]["FP_3"], all_results["mle"]["FN_3"])

            if weak_strong > 0:
                mcc_rates_mix_mle_2_weak[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP_2_weak"],
                                                                       all_results["mle"]["TN_2_weak"],
                                                                       all_results["mle"]["FP_2_weak"],
                                                                       all_results["mle"]["FN_2_weak"])
                mcc_rates_mix_mle_3_weak[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP_3_weak"],
                                                                       all_results["mle"]["TN_3_weak"],
                                                                       all_results["mle"]["FP_3_weak"],
                                                                       all_results["mle"]["FN_3_weak"])
                mcc_rates_mix_mle_weak[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP_weak"],
                                                                     all_results["mle"]["TN_weak"],
                                                                     all_results["mle"]["FP_weak"],
                                                                     all_results["mle"]["FN_weak"])
                mcc_rates_mix_mle_2_strong[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP_2_strong"],
                                                                         all_results["mle"]["TN_2_strong"],
                                                                         all_results["mle"]["FP_2_strong"],
                                                                         all_results["mle"]["FN_2_strong"])
                mcc_rates_mix_mle_3_strong[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP_3_strong"],
                                                                         all_results["mle"]["TN_3_strong"],
                                                                         all_results["mle"]["FP_3_strong"],
                                                                         all_results["mle"]["FN_3_strong"])
                mcc_rates_mix_mle_strong[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP_strong"],
                                                                       all_results["mle"]["TN_strong"],
                                                                       all_results["mle"]["FP_strong"],
                                                                       all_results["mle"]["FN_strong"])
        if all_results["lasso"] is not None and mcc:
            mcc_rates_mix_lasso[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP"], all_results["lasso"]["TN"],
                                                              all_results["lasso"]["FP"], all_results["lasso"]["FN"])
            mcc_rates_mix_lasso_2[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP_2"],
                                                                all_results["lasso"]["TN_2"],
                                                                all_results["lasso"]["FP_2"],
                                                                all_results["lasso"]["FN_2"])
            mcc_rates_mix_lasso_3[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP_3"],
                                                                all_results["lasso"]["TN_3"],
                                                                all_results["lasso"]["FP_3"],
                                                                all_results["lasso"]["FN_3"])

            if weak_strong > 0:
                mcc_rates_mix_lasso_2_weak[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP_2_weak"],
                                                                         all_results["lasso"]["TN_2_weak"],
                                                                         all_results["lasso"]["FP_2_weak"],
                                                                         all_results["lasso"]["FN_2_weak"])
                mcc_rates_mix_lasso_3_weak[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP_3_weak"],
                                                                         all_results["lasso"]["TN_3_weak"],
                                                                         all_results["lasso"]["FP_3_weak"],
                                                                         all_results["lasso"]["FN_3_weak"])
                mcc_rates_mix_lasso_weak[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP_weak"],
                                                                       all_results["lasso"]["TN_weak"],
                                                                       all_results["lasso"]["FP_weak"],
                                                                       all_results["lasso"]["FN_weak"])
                mcc_rates_mix_lasso_2_strong[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP_2_strong"],
                                                                           all_results["lasso"]["TN_2_strong"],
                                                                           all_results["lasso"]["FP_2_strong"],
                                                                           all_results["lasso"]["FN_2_strong"])
                mcc_rates_mix_lasso_3_strong[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP_3_strong"],
                                                                           all_results["lasso"]["TN_3_strong"],
                                                                           all_results["lasso"]["FP_3_strong"],
                                                                           all_results["lasso"]["FN_3_strong"])
                mcc_rates_mix_lasso_strong[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP_strong"],
                                                                         all_results["lasso"]["TN_strong"],
                                                                         all_results["lasso"]["FP_strong"],
                                                                         all_results["lasso"]["FN_strong"])

        if all_results["ada"] is not None and mcc:
            mcc_rates_mix_ada[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP"], all_results["ada"]["TN"],
                                                            all_results["ada"]["FP"], all_results["ada"]["FN"])
            mcc_rates_mix_ada_2[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP_2"], all_results["ada"]["TN_2"],
                                                              all_results["ada"]["FP_2"], all_results["ada"]["FN_2"])
            mcc_rates_mix_ada_3[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP_3"], all_results["ada"]["TN_3"],
                                                              all_results["ada"]["FP_3"], all_results["ada"]["FN_3"])

            if weak_strong > 0:
                mcc_rates_mix_ada_2_weak[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP_2_weak"],
                                                                       all_results["ada"]["TN_2_weak"],
                                                                       all_results["ada"]["FP_2_weak"],
                                                                       all_results["ada"]["FN_2_weak"])
                mcc_rates_mix_ada_3_weak[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP_3_weak"],
                                                                       all_results["ada"]["TN_3_weak"],
                                                                       all_results["ada"]["FP_3_weak"],
                                                                       all_results["ada"]["FN_3_weak"])
                mcc_rates_mix_ada_weak[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP_weak"],
                                                                     all_results["ada"]["TN_weak"],
                                                                     all_results["ada"]["FP_weak"],
                                                                     all_results["ada"]["FN_weak"])
                mcc_rates_mix_ada_2_strong[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP_2_strong"],
                                                                         all_results["ada"]["TN_2_strong"],
                                                                         all_results["ada"]["FP_2_strong"],
                                                                         all_results["ada"]["FN_2_strong"])
                mcc_rates_mix_ada_3_strong[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP_3_strong"],
                                                                         all_results["ada"]["TN_3_strong"],
                                                                         all_results["ada"]["FP_3_strong"],
                                                                         all_results["ada"]["FN_3_strong"])
                mcc_rates_mix_ada_strong[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP_strong"],
                                                                       all_results["ada"]["TN_strong"],
                                                                       all_results["ada"]["FP_strong"],
                                                                       all_results["ada"]["FN_strong"])

        if r2:
            mse_mix_mle, r2_mix_mle, pcc_mix_mle = model_lst[2][j].mse_and_r2_combined(mle_or_ols_results,
                                                                             remove_tail=remove_tail_r2,
                                                                             tp_only=tp_only)
            mse_mix_lasso, r2_mix_lasso, pcc_mix_lasso = model_lst[2][j].mse_and_r2_combined(lasso_results,
                                                                                 remove_tail=remove_tail_r2,
                                                                                 tp_only=tp_only)
            mse_mix_ada, r2_mix_ada, pcc_mix_ada = model_lst[2][j].mse_and_r2_combined(ada_results, remove_tail=remove_tail_r2,
                                                                             tp_only=tp_only)
            mse_mix_mle_arr = np.append(mse_mix_mle_arr, mse_mix_mle)
            mse_mix_lasso_arr = np.append(mse_mix_lasso_arr, mse_mix_lasso)
            mse_mix_ada_arr = np.append(mse_mix_ada_arr, mse_mix_ada)
            R2_mix_mle_arr = np.append(R2_mix_mle_arr, r2_mix_mle)
            R2_mix_lasso_arr = np.append(R2_mix_lasso_arr, r2_mix_lasso)
            R2_mix_ada_arr = np.append(R2_mix_ada_arr, r2_mix_ada)
            pcc_mix_mle_arr = np.append(pcc_mix_mle_arr, pcc_mix_mle)
            pcc_mix_lasso_arr = np.append(pcc_mix_lasso_arr, pcc_mix_lasso)
            pcc_mix_ada_arr = np.append(pcc_mix_ada_arr, pcc_mix_ada)

    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey=True, figsize=(18, 17), dpi=100)

    add_identity(ax[0, 0], color='k', ls='--')
    demo_2_0_copy, ada_2_0_copy = shrink_two_arr(demo_2_0, ada_2_0)
    demo_3_0_copy, ada_3_0_copy = shrink_two_arr(demo_3_0, ada_3_0)
    ax[0, 0].scatter(demo_2_0_copy, ada_2_0_copy, c='tab:red', marker='o', alpha=0.8, edgecolors='none')
    ax[0, 0].scatter(demo_3_0_copy, ada_3_0_copy, c='tab:red', marker='^', alpha=0.4, edgecolors='none')
    ax[0, 0].set_xlim(-0.1, 0.35)
    ax[0, 0].set_ylim(-0.1, 0.35)
    ax[0, 0].axhline(0, linestyle='--')
    ax[0, 0].axvline(0, linestyle='--')
    if r2 and mcc:
        # ax[0, 0].legend(labels=["Real", f"Adaptive LASSO, $R^2 = {round(np.average(R2_2_ada_arr), 4)}$, "
        #                                 f"$MCC = {round(np.average(mcc_rates_2_ada), 4)}$, "
        #                                 f"$MCC_2 = {round(np.average(mcc_rates_2_ada_2), 4)}$, "
        #                                 f"$MCC_3 = {round(np.average(mcc_rates_2_ada_3), 4)}$"],
        #                 loc="lower right")
        ax[0, 0].legend(labels=["Real", f"Adaptive LASSO, $R^2 = {round(np.average(R2_2_ada_arr), 4)}$, "
                                        f"$PCC = {round(np.average(pcc_2_ada_arr), 4)}$, "
                                        f"$MCC = {round(np.average(mcc_rates_2_ada), 4)}$"],
                        loc="lower right")
        print("PCC for ALASSO 2 is:", np.average(pcc_2_ada_arr))
    else:
        ax[0, 0].legend(labels=["Real", "OLS"], loc="lower right")

    add_identity(ax[0, 1], color='k', ls='--')
    demo_2_0_copy, lasso_2_0_copy = shrink_two_arr(demo_2_0, lasso_2_0)
    demo_3_0_copy, lasso_3_0_copy = shrink_two_arr(demo_3_0, lasso_3_0)
    ax[0, 1].scatter(demo_2_0_copy, lasso_2_0_copy, c='tab:cyan', marker='o', alpha=0.8, edgecolors='none')
    ax[0, 1].scatter(demo_3_0_copy, lasso_3_0_copy, c='tab:cyan', marker='^', alpha=0.4, edgecolors='none')
    ax[0, 1].set_xlim(-0.1, 0.35)
    ax[0, 1].set_ylim(-0.1, 0.35)
    ax[0, 1].axhline(0, linestyle='--')
    ax[0, 1].axvline(0, linestyle='--')
    if r2 and mcc:
        # ax[0, 1].legend(labels=["Real", f"LASSO, $R^2 = {round(np.average(R2_2_lasso_arr), 4)}$, "
        #                                 f"$MCC = {round(np.average(mcc_rates_2_lasso), 4)}$, "
        #                                 f"$MCC_2 = {round(np.average(mcc_rates_2_lasso_2), 4)}$, "
        #                                 f"$MCC_3 = {round(np.average(mcc_rates_2_lasso_3), 4)}$"],
        #                 loc="lower right")
        ax[0, 1].legend(labels=["Real", f"LASSO, $R^2 = {round(np.average(R2_2_lasso_arr), 4)}$, "
                                        f"$PCC = {round(np.average(pcc_2_lasso_arr), 4)}$, "
                                        f"$MCC = {round(np.average(mcc_rates_2_lasso), 4)}$"],
                        loc="lower right")
        print("PCC for LASSO 2 is:", np.average(pcc_2_lasso_arr))
    else:
        ax[0, 1].legend(labels=["Real", "LASSO"], loc="lower right")

    add_identity(ax[0, 2], color='k', ls='--')
    demo_2_0_copy, mle_2_0_copy = shrink_two_arr(demo_2_0, mle_2_0)
    demo_3_0_copy, mle_3_0_copy = shrink_two_arr(demo_3_0, mle_3_0)
    ax[0, 2].scatter(demo_2_0_copy, mle_2_0_copy, c='tab:purple', marker='o', alpha=0.8, edgecolors='none')
    ax[0, 2].scatter(demo_3_0_copy, mle_3_0_copy, c='tab:purple', marker='^', alpha=0.4, edgecolors='none')
    ax[0, 2].set_xlim(-0.1, 0.35)
    ax[0, 2].set_ylim(-0.1, 0.35)
    ax[0, 2].axhline(0, linestyle='--')
    ax[0, 2].axvline(0, linestyle='--')
    if r2 and mcc:
        # ax[0, 2].legend(labels=["Real", f"OLS, $R^2 = {round(np.average(R2_2_mle_arr), 4)}$, "
        #                                 f"$MCC = {round(np.average(mcc_rates_2_mle), 4)}$, "
        #                                 f"$MCC_2 = {round(np.average(mcc_rates_2_mle_2), 4)}$, "
        #                                 f"$MCC_3 = {round(np.average(mcc_rates_2_mle_3), 4)}$"],
        #                 loc="lower right")
        ax[0, 2].legend(labels=["Real", f"OLS, $R^2 = {round(np.average(R2_2_mle_arr), 4)}$, "
                                        f"$PCC = {round(np.average(pcc_2_mle_arr), 4)}$, "
                                        f"$MCC = {round(np.average(mcc_rates_2_mle), 4)}$"],
                        loc="lower right")
        print("PCC for OLS 2 is:", np.average(pcc_2_mle_arr))
    else:
        ax[0, 2].legend(labels=["Real", "Adaptive LASSO"], loc="lower right")

    add_identity(ax[1, 0], color='k', ls='--')
    demo_2_1_copy, ada_2_1_copy = shrink_two_arr(demo_2_1, ada_2_1)
    demo_3_1_copy, ada_3_1_copy = shrink_two_arr(demo_3_1, ada_3_1)
    ax[1, 0].scatter(demo_2_1_copy, ada_2_1_copy, c='tab:red', marker='o', alpha=0.8, edgecolors='none')
    ax[1, 0].scatter(demo_3_1_copy, ada_3_1_copy, c='tab:red', marker='^', alpha=0.4, edgecolors='none')
    ax[1, 0].set_xlim(-0.1, 0.35)
    ax[1, 0].set_ylim(-0.1, 0.35)
    ax[1, 0].axhline(0, linestyle='--')
    ax[1, 0].axvline(0, linestyle='--')
    if r2 and mcc:
        # ax[1, 0].legend(labels=["Real", f"Adaptive LASSO, $R^2 = {round(np.average(R2_3_ada_arr), 4)}$, "
        #                                 f"$MCC = {round(np.average(mcc_rates_3_ada), 4)}$, "
        #                                 f"$MCC_2 = {round(np.average(mcc_rates_3_ada_2), 4)}$, "
        #                                 f"$MCC_3 = {round(np.average(mcc_rates_3_ada_3), 4)}$"],
        #                 loc="lower right")
        ax[1, 0].legend(labels=["Real", f"Adaptive LASSO, $R^2 = {round(np.average(R2_3_ada_arr), 4)}$, "
                                        f"$PCC = {round(np.average(pcc_3_ada_arr), 4)}$, "
                                        f"$MCC = {round(np.average(mcc_rates_3_ada), 4)}$"],
                        loc="lower right")
        print("PCC for ALASSO 3 is:", np.average(pcc_3_ada_arr))
    else:
        ax[1, 0].legend(labels=["Real", "OLS"], loc="lower right")

    add_identity(ax[1, 1], color='k', ls='--')
    demo_2_1_copy, lasso_2_1_copy = shrink_two_arr(demo_2_1, lasso_2_1)
    demo_3_1_copy, lasso_3_1_copy = shrink_two_arr(demo_3_1, lasso_3_1)
    ax[1, 1].scatter(demo_2_1_copy, lasso_2_1_copy, c='tab:cyan', marker='o', alpha=0.8, edgecolors='none')
    ax[1, 1].scatter(demo_3_1_copy, lasso_3_1_copy, c='tab:cyan', marker='^', alpha=0.4, edgecolors='none')
    ax[1, 1].set_xlim(-0.1, 0.35)
    ax[1, 1].set_ylim(-0.1, 0.35)
    ax[1, 1].axhline(0, linestyle='--')
    ax[1, 1].axvline(0, linestyle='--')
    if r2 and mcc:
        # ax[1, 1].legend(labels=["Real", f"LASSO, $R^2 = {round(np.average(R2_3_lasso_arr), 4)}$, "
        #                                 f"$MCC = {round(np.average(mcc_rates_3_lasso), 4)}$, "
        #                                 f"$MCC_2 = {round(np.average(mcc_rates_3_lasso_2), 4)}$, "
        #                                 f"$MCC_3 = {round(np.average(mcc_rates_3_lasso_3), 4)}$"],
        #                 loc="lower right")
        ax[1, 1].legend(labels=["Real", f"LASSO, $R^2 = {round(np.average(R2_3_lasso_arr), 4)}$, "
                                        f"$PCC = {round(np.average(pcc_3_lasso_arr), 4)}$, "
                                        f"$MCC = {round(np.average(mcc_rates_3_lasso), 4)}$"],
                        loc="lower right")
        print("PCC for LASSO 3 is:", np.average(pcc_3_lasso_arr))
    else:
        ax[1, 1].legend(labels=["Real", "LASSO"], loc="lower right")

    add_identity(ax[1, 2], color='k', ls='--')
    demo_2_1_copy, mle_2_1_copy = shrink_two_arr(demo_2_1, mle_2_1)
    demo_3_1_copy, mle_3_1_copy = shrink_two_arr(demo_3_1, mle_3_1)
    ax[1, 2].scatter(demo_2_1_copy, mle_2_1_copy, c='tab:purple', marker='o', alpha=0.8, edgecolors='none')
    ax[1, 2].scatter(demo_3_1_copy, mle_3_1_copy, c='tab:purple', marker='^', alpha=0.4, edgecolors='none')
    ax[1, 2].set_xlim(-0.1, 0.35)
    ax[1, 2].set_ylim(-0.1, 0.35)
    ax[1, 2].axhline(0, linestyle='--')
    ax[1, 2].axvline(0, linestyle='--')
    if r2 and mcc:
        # ax[1, 2].legend(labels=["Real", f"OLS, $R^2 = {round(np.average(R2_3_mle_arr), 4)}$, "
        #                                 f"$MCC = {round(np.average(mcc_rates_3_mle), 4)}$, "
        #                                 f"$MCC_2 = {round(np.average(mcc_rates_3_mle_2), 4)}$, "
        #                                 f"$MCC_3 = {round(np.average(mcc_rates_3_mle_3), 4)}$"],
        #                 loc="lower right")
        ax[1, 2].legend(labels=["Real", f"OLS, $R^2 = {round(np.average(R2_3_mle_arr), 4)}$, "
                                        f"$PCC = {round(np.average(pcc_3_mle_arr), 4)}$, "
                                        f"$MCC = {round(np.average(mcc_rates_3_mle), 4)}$"],
                        loc="lower right")
        print("PCC for OLS 3 is:", np.average(pcc_3_mle_arr))
    else:
        ax[1, 2].legend(labels=["Real", "Adaptive LASSO"], loc="lower right")

    add_identity(ax[2, 0], color='k', ls='--')
    demo_2_2_copy, ada_2_2_copy = shrink_two_arr(demo_2_2, ada_2_2)
    demo_3_2_copy, ada_3_2_copy = shrink_two_arr(demo_3_2, ada_3_2)
    ax[2, 0].scatter(demo_2_2_copy, ada_2_2_copy, c='tab:red', marker='o', alpha=0.8, edgecolors='none')
    ax[2, 0].scatter(demo_3_2_copy, ada_3_2_copy, c='tab:red', marker='^', alpha=0.4, edgecolors='none')
    ax[2, 0].set_xlim(-0.1, 0.35)
    ax[2, 0].set_ylim(-0.1, 0.35)
    ax[2, 0].axhline(0, linestyle='--')
    ax[2, 0].axvline(0, linestyle='--')
    if r2 and mcc:
        # ax[2, 0].legend(labels=["Real", f"Adaptive LASSO, $R^2 = {round(np.average(R2_mix_ada_arr), 4)}$, "
        #                                 f"$MCC = {round(np.average(mcc_rates_mix_ada), 4)}$, "
        #                                 f"$MCC_2 = {round(np.average(mcc_rates_mix_ada_2), 4)}$, "
        #                                 f"$MCC_3 = {round(np.average(mcc_rates_mix_ada_3), 4)}$"], loc="lower right")
        ax[2, 0].legend(labels=["Real", f"Adaptive LASSO, $R^2 = {round(np.average(R2_mix_ada_arr), 4)}$, "
                                        f"$PCC = {round(np.average(pcc_mix_ada_arr), 4)}$, "
                                        f"$MCC = {round(np.average(mcc_rates_mix_ada), 4)}$"], loc="lower right")
        print("PCC for ALASSO mix is:", np.average(pcc_mix_ada_arr))
    else:
        ax[2, 0].legend(labels=["Real", "OLS"], loc="lower right")

    add_identity(ax[2, 1], color='k', ls='--')
    demo_2_2_copy, lasso_2_2_copy = shrink_two_arr(demo_2_2, lasso_2_2)
    demo_3_2_copy, lasso_3_2_copy = shrink_two_arr(demo_3_2, lasso_3_2)
    ax[2, 1].scatter(demo_2_2_copy, lasso_2_2_copy, c='tab:cyan', marker='o', alpha=0.8, edgecolors='none')
    ax[2, 1].scatter(demo_3_2_copy, lasso_3_2_copy, c='tab:cyan', marker='^', alpha=0.4, edgecolors='none')
    ax[2, 1].set_xlim(-0.1, 0.35)
    ax[2, 1].set_ylim(-0.1, 0.35)
    ax[2, 1].axhline(0, linestyle='--')
    ax[2, 1].axvline(0, linestyle='--')
    if r2 and mcc:
        # ax[2, 1].legend(labels=["Real", f"LASSO, $R^2 = {round(np.average(R2_mix_lasso_arr), 4)}$, "
        #                                 f"$MCC = {round(np.average(mcc_rates_mix_lasso), 4)}$, "
        #                                 f"$MCC_2 = {round(np.average(mcc_rates_mix_lasso_2), 4)}$, "
        #                                 f"$MCC_3 = {round(np.average(mcc_rates_mix_lasso_3), 4)}$"], loc="lower right")
        ax[2, 1].legend(labels=["Real", f"LASSO, $R^2 = {round(np.average(R2_mix_lasso_arr), 4)}$, "
                                        f"$PCC = {round(np.average(pcc_mix_lasso_arr), 4)}$, "
                                        f"$MCC = {round(np.average(mcc_rates_mix_lasso), 4)}$"], loc="lower right")
        print("PCC for LASSO mix is:", np.average(pcc_mix_lasso_arr))
    else:
        ax[2, 1].legend(labels=["Real", "LASSO"], loc="lower right")

    add_identity(ax[2, 2], color='k', ls='--')
    demo_2_2_copy, mle_2_2_copy = shrink_two_arr(demo_2_2, mle_2_2)
    demo_3_2_copy, mle_3_2_copy = shrink_two_arr(demo_3_2, mle_3_2)
    ax[2, 2].scatter(demo_2_2_copy, mle_2_2_copy, c='tab:purple', marker='o', alpha=0.8, edgecolors='none')
    ax[2, 2].scatter(demo_3_2_copy, mle_3_2_copy, c='tab:purple', marker='^', alpha=0.4, edgecolors='none')
    ax[2, 2].set_xlim(-0.1, 0.35)
    ax[2, 2].set_ylim(-0.1, 0.35)
    ax[2, 2].axhline(0, linestyle='--')
    ax[2, 2].axvline(0, linestyle='--')
    if r2 and mcc:
        # ax[2, 2].legend(labels=["Real", f"OLS, $R^2 = {round(np.average(R2_mix_mle_arr), 4)}$, "
        #                                 f"$MCC = {round(np.average(mcc_rates_mix_mle), 4)}$, "
        #                                 f"$MCC_2 = {round(np.average(mcc_rates_mix_mle_2), 4)}$, "
        #                                 f"$MCC_3 = {round(np.average(mcc_rates_mix_mle_3), 4)}$"],
        #                 loc="lower right")
        ax[2, 2].legend(labels=["Real", f"OLS, $R^2 = {round(np.average(R2_mix_mle_arr), 4)}$, "
                                        f"$PCC = {round(np.average(pcc_mix_mle_arr), 4)}$, "
                                        f"$MCC = {round(np.average(mcc_rates_mix_mle), 4)}$"],
                        loc="lower right")
        # print("PCC for OLS mix is:", np.average(pcc_mix_mle_arr))
    else:
        ax[2, 2].legend(labels=["Real", "OLS"], loc="lower right")

    for i in range(3):
        for j in range(3):
            ax[i, j].spines[['right', 'top']].set_visible(False)

    ax[0, 0].set_ylabel('Pure Pairwise \n Coupling Strength', rotation=90, fontsize=24)
    ax[1, 0].set_ylabel('Pure 3-interaction \n Coupling Strength', rotation=90, fontsize=24)
    ax[2, 0].set_ylabel('Mixture \n Coupling Strength', rotation=90, fontsize=24)
    ax[0, 0].tick_params(axis='y', labelsize=18)
    ax[1, 0].tick_params(axis='y', labelsize=18)
    ax[2, 0].tick_params(axis='y', labelsize=18)

    title_1 = f"Adaptive LASSO"
    title_2 = f"LASSO"
    title_3 = f"OLS"
    ax[0, 0].set_title(title_1, fontsize=30)
    ax[0, 1].set_title(title_2, fontsize=30)
    ax[0, 2].set_title(title_3, fontsize=30)

    # plt.rcParams["font.size"] = 36  # フォントを大きく
    xlabel = 'Coupling strength'
    fig.suptitle('Coupling strength scatter plot', size=40)
    params = {'legend.fontsize': 20,
              'legend.handlelength': 1}
    plt.rcParams.update(params)

    hori_arr = np.array([-0.1, 0, 0.1, 0.2, 0.3])

    ax[-1, 0].set_xlabel(xlabel, fontsize=24)
    ax[-1, 0].set_xticks(hori_arr)
    ax[-1, 0].tick_params(axis='x', labelsize=18)
    ax[-1, 0].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    ax[-1, 1].set_xlabel(xlabel, fontsize=24)
    ax[-1, 1].set_xticks(hori_arr)
    ax[-1, 1].tick_params(axis='x', labelsize=18)
    ax[-1, 1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    ax[-1, 2].set_xlabel(xlabel, fontsize=24)
    ax[-1, 2].set_xticks(hori_arr)
    ax[-1, 2].tick_params(axis='x', labelsize=18)
    ax[-1, 2].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    path_base = make_dirs(p_)

    path = path_base + "/coupling.jpg"
    fig.savefig(path)
    path_eps = path_base + "/coupling.eps"
    fig.savefig(path_eps, format="eps", bbox_inches=None)
    plt.show()

    mcc_csv = path_base + "/MCC_Fig4.csv"
    mse_csv = path_base + "/mse_Fig4.csv"
    seeds_csv = path_base + f"/seeds_for_{conn_seed_}.csv"

    if r2 and mcc:
        with open(mse_csv, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header22 = ["", "Adaptive LASSO", "LASSO", "OLS with FDR control"]
            writer.writerow(header22)
            first_row = ["Pairwise", np.average(mse_2_ada_arr), np.average(mse_2_lasso_arr), np.average(mse_2_mle_arr)]
            writer.writerow(first_row)
            second_row = ["3-interaction", np.average(mse_3_ada_arr), np.average(mse_3_lasso_arr),
                          np.average(mse_3_mle_arr)]
            writer.writerow(second_row)
            third_row = ["Mixture", np.average(mse_mix_ada_arr), np.average(mse_mix_lasso_arr),
                          np.average(mse_mix_mle_arr)]
            writer.writerow(third_row)
            f.close()

        with open(mcc_csv, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header2 = ["", "Adaptive LASSO", "LASSO", "OLS with FDR control",
                       "Adaptive LASSO - 2", "LASSO - 2", "OLS with FDR control - 2",
                       "Adaptive LASSO - 3", "LASSO - 3", "OLS with FDR control - 3",
                       "Adaptive LASSO - 2 - weak", "LASSO - 2 - weak", "OLS with FDR control - 2 - weak",
                       "Adaptive LASSO - 3 - weak", "LASSO - 3 - weak", "OLS with FDR control - 3 - weak",
                       "Adaptive LASSO - 2 - strong", "LASSO - 2 - strong", "OLS with FDR control - 2 - strong",
                       "Adaptive LASSO - 3 - strong", "LASSO - 3 - strong", "OLS with FDR control - 3 - strong",
                       "Adaptive LASSO - weak", "LASSO - weak", "OLS with FDR control - weak",
                       "Adaptive LASSO - strong", "LASSO - strong", "OLS with FDR control - strong",
                       ]
            writer.writerow(header2)
            first_row = ["Pairwise", np.average(mcc_rates_2_ada), np.average(mcc_rates_2_lasso),
                         np.average(mcc_rates_2_mle), np.average(mcc_rates_2_ada_2), np.average(mcc_rates_2_lasso_2),
                         np.average(mcc_rates_2_mle_2), np.average(mcc_rates_2_ada_3), np.average(mcc_rates_2_lasso_3),
                         np.average(mcc_rates_2_mle_3), np.average(mcc_rates_2_ada_2_weak),
                         np.average(mcc_rates_2_lasso_2_weak),
                         np.average(mcc_rates_2_mle_2_weak), np.average(mcc_rates_2_ada_3_weak),
                         np.average(mcc_rates_2_lasso_3_weak),
                         np.average(mcc_rates_2_mle_3_weak), np.average(mcc_rates_2_ada_2_strong),
                         np.average(mcc_rates_2_lasso_2_strong),
                         np.average(mcc_rates_2_mle_2_strong), np.average(mcc_rates_2_ada_3_strong),
                         np.average(mcc_rates_2_lasso_3_strong),
                         np.average(mcc_rates_2_mle_3_strong), np.average(mcc_rates_2_ada_weak),
                         np.average(mcc_rates_2_lasso_weak),
                         np.average(mcc_rates_2_mle_weak), np.average(mcc_rates_2_ada_strong),
                         np.average(mcc_rates_2_lasso_strong),
                         np.average(mcc_rates_2_mle_strong)]
            sec_row = ["3-interaction", np.average(mcc_rates_3_ada), np.average(mcc_rates_3_lasso),
                       np.average(mcc_rates_3_mle), np.average(mcc_rates_3_ada_2), np.average(mcc_rates_3_lasso_2),
                       np.average(mcc_rates_3_mle_2), np.average(mcc_rates_3_ada_3), np.average(mcc_rates_3_lasso_3),
                       np.average(mcc_rates_3_mle_3), np.average(mcc_rates_3_ada_2_weak),
                       np.average(mcc_rates_3_lasso_2_weak),
                       np.average(mcc_rates_3_mle_2_weak), np.average(mcc_rates_3_ada_3_weak),
                       np.average(mcc_rates_3_lasso_3_weak),
                       np.average(mcc_rates_3_mle_3_weak), np.average(mcc_rates_3_ada_2_strong),
                       np.average(mcc_rates_3_lasso_2_strong),
                       np.average(mcc_rates_3_mle_2_strong), np.average(mcc_rates_3_ada_3_strong),
                       np.average(mcc_rates_3_lasso_3_strong),
                       np.average(mcc_rates_3_mle_3_strong), np.average(mcc_rates_3_ada_weak),
                       np.average(mcc_rates_3_lasso_weak),
                       np.average(mcc_rates_3_mle_weak), np.average(mcc_rates_3_ada_strong),
                       np.average(mcc_rates_3_lasso_strong),
                       np.average(mcc_rates_3_mle_strong)]
            third_row = ["Mixture", np.average(mcc_rates_mix_ada), np.average(mcc_rates_mix_lasso),
                         np.average(mcc_rates_mix_mle), np.average(mcc_rates_mix_ada_2),
                         np.average(mcc_rates_mix_lasso_2),
                         np.average(mcc_rates_mix_mle_2), np.average(mcc_rates_mix_ada_3),
                         np.average(mcc_rates_mix_lasso_3),
                         np.average(mcc_rates_mix_mle_3), np.average(mcc_rates_mix_ada_2_weak),
                         np.average(mcc_rates_mix_lasso_2_weak),
                         np.average(mcc_rates_mix_mle_2_weak), np.average(mcc_rates_mix_ada_3_weak),
                         np.average(mcc_rates_mix_lasso_3_weak),
                         np.average(mcc_rates_mix_mle_3_weak), np.average(mcc_rates_mix_ada_2_strong),
                         np.average(mcc_rates_mix_lasso_2_strong),
                         np.average(mcc_rates_mix_mle_2_strong), np.average(mcc_rates_mix_ada_3_strong),
                         np.average(mcc_rates_mix_lasso_3_strong),
                         np.average(mcc_rates_mix_mle_3_strong), np.average(mcc_rates_mix_ada_weak),
                         np.average(mcc_rates_mix_lasso_weak),
                         np.average(mcc_rates_mix_mle_weak), np.average(mcc_rates_mix_ada_strong),
                         np.average(mcc_rates_mix_lasso_strong),
                         np.average(mcc_rates_mix_mle_strong)]
            writer.writerow(first_row)
            writer.writerow(sec_row)
            writer.writerow(third_row)

    with open(seeds_csv, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(real_2_lst)
        writer.writerow(real_3_lst)

    return real_2_lst, real_3_lst


if __name__ == "__main__":
    natfreq_seed = 98765
    conn_seed = 19961102
    coup_seed = 976567
    noise_seed = 20116991

    start_datetime = datetime.now()
    p = 0.1

    my_data = np.genfromtxt("fig4_seeds.csv", delimiter=',')
    two_arr = np.copy(my_data[0, :]).astype(int)
    three_arr = np.copy(my_data[1, :]).astype(int)

    # two_seed_arr, three_seed_arr = main_for_Fig4(natfreq_seed, conn_seed, coup_seed, noise_seed,
    #                                              pre_conn_2_arr=None, pre_conn_3_arr=None, p_=p, r2=True, mcc=True)
    two_seed_arr, three_seed_arr = main_for_Fig4(natfreq_seed, conn_seed, coup_seed, noise_seed,
                                                 pre_conn_2_arr=two_arr, pre_conn_3_arr=three_arr, p_=p, r2=True,
                                                 mcc=True, weak_strong=0.1)
    print(two_seed_arr)
    print(three_seed_arr)

    now = datetime.now()
    duration = now - start_datetime
    print("Duration is =", duration)
