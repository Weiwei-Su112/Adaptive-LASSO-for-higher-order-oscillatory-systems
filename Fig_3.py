from GeneralInteraction_ori import *
from matplotlib.ticker import PercentFormatter


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time}")
        return result

    return wrapper


def make_dirs(p_, K_or_T):
    now_ = datetime.now()
    date = now_.strftime("%Y%m%d")
    tm_string = now_.strftime("%H%M%S")
    file_path1 = "Fig3data/img/" + date + "/"
    os.makedirs(file_path1, exist_ok=True)

    if K_or_T == 1:
        addon = "_K/"
    else:
        addon = "_T/"
    file_path_p = "Fig3data/img/" + date + "/" + tm_string + f"_p={str(p_)}" + addon
    os.mkdir(file_path_p)
    return file_path_p


def make_dirs_for_mcc(p_lst_, K_or_T):
    now_ = datetime.now()
    date = now_.strftime("%Y%m%d")
    tm_string = now_.strftime("%H%M%S")
    file_path1 = "Fig3data/img/" + date + "/"
    os.makedirs(file_path1, exist_ok=True)

    file_path_base = "Fig3data/img/" + date + "/" + tm_string + "/"
    os.makedirs(file_path_base, exist_ok=True)

    if K_or_T == 1:
        addon = "_K/"
    else:
        addon = "_T/"
    file_path_p_lst = []
    for p in p_lst_:
        file_path_p = file_path_base + f"p={str(p)}" + addon
        os.mkdir(file_path_p)
        file_path_p_lst.append(file_path_p)
    return file_path_base, file_path_p_lst


def main_for_Fig3(natfreq_seed_, conn_seed_, coup_seed_, noise_seed_, K_or_T=1, trial_num=5,
                  pre_conn_2_lst=None, pre_conn_3_lst=None, p_lst_=[0.05, 0.1, 0.15], draw=2, mle_ori=False, ols=True,
                  lasso=True, ada=True, mle_threshold=0.1):
    if not ols:
        assert not mle_ori, "Need OLS results for doing threshold operation. "
    # 1. uni parameters
    scan_num = 8
    n_nodes = 12

    dt = 0.02
    noise_sth = 0.2
    st_fr = 1 / 9
    inf_l = 2 / 3
    # filter variable
    ops_limit = 0.4
    reduce_seed = 19374759102
    ratio = 6.0
    all_connected = False

    if draw == 2:
        outputs = []
        file_path_base, file_path_p_lst = make_dirs_for_mcc(p_lst_, K_or_T)

    for ir in range(len(p_lst_)):
        p_ = p_lst_[ir]
        if draw == 1:
            paths = make_dirs(p_, K_or_T)
        elif draw == 2:
            paths = file_path_p_lst[ir]

        # 2. variable (K or T) (1 is K, 2 is T)
        assert K_or_T == 1 or K_or_T == 2, "K_or_T can only be 1 or 2. 1 is K and 2 is T. "
        natfreq_arr = np.random.default_rng(natfreq_seed_).normal(loc=1, scale=0.1, size=n_nodes)
        noise_arr = np.random.default_rng(noise_seed_).integers(0, 1e10, size=3)

        real_2_lst = []
        real_3_lst = []

        if K_or_T == 1:
            T = 900
            K_arr = np.linspace(0.02, 0.02 * scan_num, num=scan_num)
            hori_arr = np.copy(K_arr)

            fpr_arr_2_mle = np.zeros(scan_num)
            fnr_arr_2_mle = np.zeros(scan_num)
            mcc_arr_2_mle = np.zeros(scan_num)
            new_arr_2_mle = np.zeros(scan_num)
            fpr_arr_2_mle_ori = np.zeros(scan_num)
            fnr_arr_2_mle_ori = np.zeros(scan_num)
            mcc_arr_2_mle_ori = np.zeros(scan_num)
            new_arr_2_mle_ori = np.zeros(scan_num)
            fpr_arr_2_lasso = np.zeros(scan_num)
            fnr_arr_2_lasso = np.zeros(scan_num)
            mcc_arr_2_lasso = np.zeros(scan_num)
            new_arr_2_lasso = np.zeros(scan_num)
            fpr_arr_2_ada = np.zeros(scan_num)
            fnr_arr_2_ada = np.zeros(scan_num)
            mcc_arr_2_ada = np.zeros(scan_num)
            new_arr_2_ada = np.zeros(scan_num)

            fpr_arr_3_mle = np.zeros(scan_num)
            fnr_arr_3_mle = np.zeros(scan_num)
            mcc_arr_3_mle = np.zeros(scan_num)
            new_arr_3_mle = np.zeros(scan_num)
            fpr_arr_3_mle_ori = np.zeros(scan_num)
            fnr_arr_3_mle_ori = np.zeros(scan_num)
            mcc_arr_3_mle_ori = np.zeros(scan_num)
            new_arr_3_mle_ori = np.zeros(scan_num)
            fpr_arr_3_lasso = np.zeros(scan_num)
            fnr_arr_3_lasso = np.zeros(scan_num)
            mcc_arr_3_lasso = np.zeros(scan_num)
            new_arr_3_lasso = np.zeros(scan_num)
            fpr_arr_3_ada = np.zeros(scan_num)
            fnr_arr_3_ada = np.zeros(scan_num)
            mcc_arr_3_ada = np.zeros(scan_num)
            new_arr_3_ada = np.zeros(scan_num)

            fpr_arr_mix_mle = np.zeros(scan_num)
            fnr_arr_mix_mle = np.zeros(scan_num)
            mcc_arr_mix_mle = np.zeros(scan_num)
            new_arr_mix_mle = np.zeros(scan_num)
            fpr_arr_mix_mle_ori = np.zeros(scan_num)
            fnr_arr_mix_mle_ori = np.zeros(scan_num)
            mcc_arr_mix_mle_ori = np.zeros(scan_num)
            new_arr_mix_mle_ori = np.zeros(scan_num)
            fpr_arr_mix_lasso = np.zeros(scan_num)
            fnr_arr_mix_lasso = np.zeros(scan_num)
            mcc_arr_mix_lasso = np.zeros(scan_num)
            new_arr_mix_lasso = np.zeros(scan_num)
            fpr_arr_mix_ada = np.zeros(scan_num)
            fnr_arr_mix_ada = np.zeros(scan_num)
            mcc_arr_mix_ada = np.zeros(scan_num)
            new_arr_mix_ada = np.zeros(scan_num)

            for i in range(scan_num):
                if pre_conn_2_lst is None:
                    pre_conn_2_arr = None
                else:
                    pre_conn_2_arr = pre_conn_2_lst[i]

                if pre_conn_3_lst is None:
                    pre_conn_3_arr = None
                else:
                    pre_conn_3_arr = pre_conn_2_lst[i]

                model_lst, real_conn_2_arr, real_conn_3_arr = create_task_Fig3(conn_seed_=conn_seed_,
                                                                               coup_seed_=coup_seed_,
                                                                               noise_arr=noise_arr,
                                                                               natfreq_arr=natfreq_arr,
                                                                               pre_conn_2_arr=pre_conn_2_arr,
                                                                               trial_num=trial_num, K=K_arr[i], T=T,
                                                                               dt=dt, p_=p_,
                                                                               noise_sth=noise_sth, ratio=ratio,
                                                                               st_fr=st_fr,
                                                                               inf_l=inf_l,
                                                                               reduce_seed=reduce_seed,
                                                                               ops_limit=ops_limit,
                                                                               all_connected=all_connected,
                                                                               pre_conn_3_arr=pre_conn_3_arr)

                if np.any(real_conn_2_arr != 0):
                    print("The outcome conn2_arr is", real_conn_2_arr)
                real_2_lst.append(real_conn_2_arr)
                if np.any(real_conn_3_arr != 0):
                    print("The outcome conn3_arr is", real_conn_3_arr)
                real_3_lst.append(real_conn_3_arr)

                assert len(model_lst) == 3, "len of model_lst has to be 3, pairwise, 3-int, and mixture. "

                fpr_rates_2_mle = np.zeros(trial_num)
                fnr_rates_2_mle = np.zeros(trial_num)
                mcc_rates_2_mle = np.zeros(trial_num)
                new_rates_2_mle = np.zeros(trial_num)
                fpr_rates_2_mle_ori = np.zeros(trial_num)
                fnr_rates_2_mle_ori = np.zeros(trial_num)
                mcc_rates_2_mle_ori = np.zeros(trial_num)
                new_rates_2_mle_ori = np.zeros(trial_num)
                fpr_rates_2_lasso = np.zeros(trial_num)
                fnr_rates_2_lasso = np.zeros(trial_num)
                mcc_rates_2_lasso = np.zeros(trial_num)
                new_rates_2_lasso = np.zeros(trial_num)
                fpr_rates_2_ada = np.zeros(trial_num)
                fnr_rates_2_ada = np.zeros(trial_num)
                mcc_rates_2_ada = np.zeros(trial_num)
                new_rates_2_ada = np.zeros(trial_num)
                for j in range(trial_num):
                    act_mat = model_lst[0][j].run()
                    prepare_all = model_lst[0][j].prepare_diffs(act_mat)
                    if ols:
                        mle_or_ols_results = model_lst[0][j].solve_ols(all_prepared=prepare_all)
                    else:
                        mle_or_ols_results = None
                    if ada:
                        ada_results = model_lst[0][j].solve_ada_lasso(all_prepared=prepare_all,
                                                                      mle_or_ols_results=mle_or_ols_results)
                    else:
                        ada_results = None
                    if lasso:
                        lasso_results = model_lst[0][j].solve_lasso(all_prepared=prepare_all)
                    else:
                        lasso_results = None
                    all_results = model_lst[0][j].conn_criteria_base(mle_or_ols_results=mle_or_ols_results,
                                                                     lasso_results=lasso_results,
                                                                     ada_results=ada_results, mle_ori=mle_ori,
                                                                     mle_threshold=mle_threshold)
                    if all_results["mle"] is not None:
                        fpr_rates_2_mle[j] = model_lst[0][j].FPR_easy(all_results["mle"]["FP"],
                                                                      all_results["mle"]["TN"])
                        fnr_rates_2_mle[j] = model_lst[0][j].FNR_easy(all_results["mle"]["FN"],
                                                                      all_results["mle"]["TP"])
                        mcc_rates_2_mle[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP"],
                                                                      all_results["mle"]["TN"],
                                                                      all_results["mle"]["FP"],
                                                                      all_results["mle"]["FN"])
                        new_rates_2_mle[j] = model_lst[0][j].new_rates_easy(all_results["mle"]["A"],
                                                                            all_results["mle"]["B"])
                    if all_results["mle_ori"] is not None:
                        fpr_rates_2_mle_ori[j] = model_lst[0][j].FPR_easy(all_results["mle_ori"]["FP"],
                                                                          all_results["mle_ori"]["TN"])
                        fnr_rates_2_mle_ori[j] = model_lst[0][j].FNR_easy(all_results["mle_ori"]["FN"],
                                                                          all_results["mle_ori"]["TP"])
                        mcc_rates_2_mle_ori[j] = model_lst[0][j].MCC_easy(all_results["mle_ori"]["TP"],
                                                                          all_results["mle_ori"]["TN"],
                                                                          all_results["mle_ori"]["FP"],
                                                                          all_results["mle_ori"]["FN"])
                        new_rates_2_mle_ori[j] = model_lst[0][j].new_rates_easy(all_results["mle_ori"]["A"],
                                                                                all_results["mle_ori"]["B"])

                    if all_results["lasso"] is not None:
                        fpr_rates_2_lasso[j] = model_lst[0][j].FPR_easy(all_results["lasso"]["FP"],
                                                                        all_results["lasso"]["TN"])
                        fnr_rates_2_lasso[j] = model_lst[0][j].FNR_easy(all_results["lasso"]["FN"],
                                                                        all_results["lasso"]["TP"])
                        mcc_rates_2_lasso[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP"],
                                                                        all_results["lasso"]["TN"],
                                                                        all_results["lasso"]["FP"],
                                                                        all_results["lasso"]["FN"])
                        new_rates_2_lasso[j] = model_lst[0][j].new_rates_easy(all_results["lasso"]["A"],
                                                                              all_results["lasso"]["B"])
                    if all_results["ada"] is not None:
                        fpr_rates_2_ada[j] = model_lst[0][j].FPR_easy(all_results["ada"]["FP"],
                                                                      all_results["ada"]["TN"])
                        fnr_rates_2_ada[j] = model_lst[0][j].FNR_easy(all_results["ada"]["FN"],
                                                                      all_results["ada"]["TP"])
                        mcc_rates_2_ada[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP"],
                                                                      all_results["ada"]["TN"],
                                                                      all_results["ada"]["FP"],
                                                                      all_results["ada"]["FN"])
                        new_rates_2_ada[j] = model_lst[0][j].new_rates_easy(all_results["ada"]["A"],
                                                                            all_results["ada"]["B"])

                fpr_arr_2_mle[i] = np.average(fpr_rates_2_mle)
                fnr_arr_2_mle[i] = np.average(fpr_rates_2_mle)
                mcc_arr_2_mle[i] = np.average(mcc_rates_2_mle)
                new_arr_2_mle[i] = np.nanmean(new_rates_2_mle)
                fpr_arr_2_mle_ori[i] = np.average(fpr_rates_2_mle_ori)
                fnr_arr_2_mle_ori[i] = np.average(fpr_rates_2_mle_ori)
                mcc_arr_2_mle_ori[i] = np.average(mcc_rates_2_mle_ori)
                new_arr_2_mle_ori[i] = np.nanmean(new_rates_2_mle_ori)
                fpr_arr_2_lasso[i] = np.average(fpr_rates_2_lasso)
                fnr_arr_2_lasso[i] = np.average(fnr_rates_2_lasso)
                mcc_arr_2_lasso[i] = np.average(mcc_rates_2_lasso)
                new_arr_2_lasso[i] = np.nanmean(new_rates_2_lasso)
                fpr_arr_2_ada[i] = np.average(fpr_rates_2_ada)
                fnr_arr_2_ada[i] = np.average(fnr_rates_2_ada)
                mcc_arr_2_ada[i] = np.average(mcc_rates_2_ada)
                new_arr_2_ada[i] = np.nanmean(new_rates_2_ada)

                fpr_rates_3_mle = np.zeros(trial_num)
                fnr_rates_3_mle = np.zeros(trial_num)
                mcc_rates_3_mle = np.zeros(trial_num)
                new_rates_3_mle = np.zeros(trial_num)
                fpr_rates_3_mle_ori = np.zeros(trial_num)
                fnr_rates_3_mle_ori = np.zeros(trial_num)
                mcc_rates_3_mle_ori = np.zeros(trial_num)
                new_rates_3_mle_ori = np.zeros(trial_num)
                fpr_rates_3_lasso = np.zeros(trial_num)
                fnr_rates_3_lasso = np.zeros(trial_num)
                mcc_rates_3_lasso = np.zeros(trial_num)
                new_rates_3_lasso = np.zeros(trial_num)
                fpr_rates_3_ada = np.zeros(trial_num)
                fnr_rates_3_ada = np.zeros(trial_num)
                mcc_rates_3_ada = np.zeros(trial_num)
                new_rates_3_ada = np.zeros(trial_num)
                for j in range(trial_num):
                    act_mat = model_lst[1][j].run()
                    prepare_all = model_lst[1][j].prepare_diffs(act_mat)
                    if ols:
                        mle_or_ols_results = model_lst[1][j].solve_ols(all_prepared=prepare_all)
                    else:
                        mle_or_ols_results = None
                    if ada:
                        ada_results = model_lst[1][j].solve_ada_lasso(all_prepared=prepare_all,
                                                                      mle_or_ols_results=mle_or_ols_results)
                    else:
                        ada_results = None
                    if lasso:
                        lasso_results = model_lst[1][j].solve_lasso(all_prepared=prepare_all)
                    else:
                        lasso_results = None
                    all_results = model_lst[1][j].conn_criteria_base(mle_or_ols_results=mle_or_ols_results,
                                                                     lasso_results=lasso_results,
                                                                     ada_results=ada_results,
                                                                     mle_ori=mle_ori, mle_threshold=mle_threshold)

                    if all_results["mle"] is not None:
                        fpr_rates_3_mle[j] = model_lst[1][j].FPR_easy(all_results["mle"]["FP"],
                                                                      all_results["mle"]["TN"])
                        fnr_rates_3_mle[j] = model_lst[1][j].FNR_easy(all_results["mle"]["FN"],
                                                                      all_results["mle"]["TP"])
                        mcc_rates_3_mle[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP"],
                                                                      all_results["mle"]["TN"],
                                                                      all_results["mle"]["FP"],
                                                                      all_results["mle"]["FN"])
                        new_rates_3_mle[j] = model_lst[1][j].new_rates_easy(all_results["mle"]["A"],
                                                                            all_results["mle"]["B"])
                    if all_results["mle_ori"] is not None:
                        fpr_rates_3_mle_ori[j] = model_lst[1][j].FPR_easy(all_results["mle_ori"]["FP"],
                                                                          all_results["mle_ori"]["TN"])
                        fnr_rates_3_mle_ori[j] = model_lst[1][j].FNR_easy(all_results["mle_ori"]["FN"],
                                                                          all_results["mle_ori"]["TP"])
                        mcc_rates_3_mle_ori[j] = model_lst[1][j].MCC_easy(all_results["mle_ori"]["TP"],
                                                                          all_results["mle_ori"]["TN"],
                                                                          all_results["mle_ori"]["FP"],
                                                                          all_results["mle_ori"]["FN"])
                        new_rates_3_mle_ori[j] = model_lst[1][j].new_rates_easy(all_results["mle_ori"]["A"],
                                                                                all_results["mle_ori"]["B"])

                    if all_results["lasso"] is not None:
                        fpr_rates_3_lasso[j] = model_lst[1][j].FPR_easy(all_results["lasso"]["FP"],
                                                                        all_results["lasso"]["TN"])
                        fnr_rates_3_lasso[j] = model_lst[1][j].FNR_easy(all_results["lasso"]["FN"],
                                                                        all_results["lasso"]["TP"])
                        mcc_rates_3_lasso[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP"],
                                                                        all_results["lasso"]["TN"],
                                                                        all_results["lasso"]["FP"],
                                                                        all_results["lasso"]["FN"])
                        new_rates_3_lasso[j] = model_lst[1][j].new_rates_easy(all_results["lasso"]["A"],
                                                                              all_results["lasso"]["B"])
                    if all_results["ada"] is not None:
                        fpr_rates_3_ada[j] = model_lst[1][j].FPR_easy(all_results["ada"]["FP"],
                                                                      all_results["ada"]["TN"])
                        fnr_rates_3_ada[j] = model_lst[1][j].FNR_easy(all_results["ada"]["FN"],
                                                                      all_results["ada"]["TP"])
                        mcc_rates_3_ada[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP"],
                                                                      all_results["ada"]["TN"],
                                                                      all_results["ada"]["FP"],
                                                                      all_results["ada"]["FN"])
                        new_rates_3_ada[j] = model_lst[1][j].new_rates_easy(all_results["ada"]["A"],
                                                                            all_results["ada"]["B"])

                fpr_arr_3_mle[i] = np.average(fpr_rates_3_mle)
                fnr_arr_3_mle[i] = np.average(fpr_rates_3_mle)
                mcc_arr_3_mle[i] = np.average(mcc_rates_3_mle)
                new_arr_3_mle[i] = np.nanmean(new_rates_3_mle)
                fpr_arr_3_mle_ori[i] = np.average(fpr_rates_3_mle_ori)
                fnr_arr_3_mle_ori[i] = np.average(fpr_rates_3_mle_ori)
                mcc_arr_3_mle_ori[i] = np.average(mcc_rates_3_mle_ori)
                new_arr_3_mle_ori[i] = np.nanmean(new_rates_3_mle_ori)
                fpr_arr_3_lasso[i] = np.average(fpr_rates_3_lasso)
                fnr_arr_3_lasso[i] = np.average(fnr_rates_3_lasso)
                mcc_arr_3_lasso[i] = np.average(mcc_rates_3_lasso)
                new_arr_3_lasso[i] = np.nanmean(new_rates_3_lasso)
                fpr_arr_3_ada[i] = np.average(fpr_rates_3_ada)
                fnr_arr_3_ada[i] = np.average(fnr_rates_3_ada)
                mcc_arr_3_ada[i] = np.average(mcc_rates_3_ada)
                new_arr_3_ada[i] = np.nanmean(new_rates_3_ada)

                fpr_rates_mix_mle = np.zeros(trial_num)
                fnr_rates_mix_mle = np.zeros(trial_num)
                mcc_rates_mix_mle = np.zeros(trial_num)
                new_rates_mix_mle = np.zeros(trial_num)
                fpr_rates_mix_mle_ori = np.zeros(trial_num)
                fnr_rates_mix_mle_ori = np.zeros(trial_num)
                mcc_rates_mix_mle_ori = np.zeros(trial_num)
                new_rates_mix_mle_ori = np.zeros(trial_num)
                fpr_rates_mix_lasso = np.zeros(trial_num)
                fnr_rates_mix_lasso = np.zeros(trial_num)
                mcc_rates_mix_lasso = np.zeros(trial_num)
                new_rates_mix_lasso = np.zeros(trial_num)
                fpr_rates_mix_ada = np.zeros(trial_num)
                fnr_rates_mix_ada = np.zeros(trial_num)
                mcc_rates_mix_ada = np.zeros(trial_num)
                new_rates_mix_ada = np.zeros(trial_num)
                for j in range(trial_num):
                    act_mat = model_lst[2][j].run()
                    prepare_all = model_lst[2][j].prepare_diffs(act_mat)
                    if ols:
                        mle_or_ols_results = model_lst[2][j].solve_ols(all_prepared=prepare_all)
                    else:
                        mle_or_ols_results = None
                    if ada:
                        ada_results = model_lst[2][j].solve_ada_lasso(all_prepared=prepare_all,
                                                                      mle_or_ols_results=mle_or_ols_results)
                    else:
                        ada_results = None
                    if lasso:
                        lasso_results = model_lst[2][j].solve_lasso(all_prepared=prepare_all)
                    else:
                        lasso_results = None
                    all_results = model_lst[2][j].conn_criteria_base(mle_or_ols_results=mle_or_ols_results,
                                                                     lasso_results=lasso_results,
                                                                     ada_results=ada_results,
                                                                     mle_ori=mle_ori, mle_threshold=mle_threshold)
                    if all_results["mle"] is not None:
                        fpr_rates_mix_mle[j] = model_lst[2][j].FPR_easy(all_results["mle"]["FP"],
                                                                        all_results["mle"]["TN"])
                        fnr_rates_mix_mle[j] = model_lst[2][j].FNR_easy(all_results["mle"]["FN"],
                                                                        all_results["mle"]["TP"])
                        mcc_rates_mix_mle[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP"],
                                                                        all_results["mle"]["TN"],
                                                                        all_results["mle"]["FP"],
                                                                        all_results["mle"]["FN"])
                        new_rates_mix_mle[j] = model_lst[2][j].new_rates_easy(all_results["mle"]["A"],
                                                                              all_results["mle"]["B"])
                    if all_results["mle_ori"] is not None:
                        fpr_rates_mix_mle_ori[j] = model_lst[2][j].FPR_easy(all_results["mle_ori"]["FP"],
                                                                            all_results["mle_ori"]["TN"])
                        fnr_rates_mix_mle_ori[j] = model_lst[2][j].FNR_easy(all_results["mle_ori"]["FN"],
                                                                            all_results["mle_ori"]["TP"])
                        mcc_rates_mix_mle_ori[j] = model_lst[2][j].MCC_easy(all_results["mle_ori"]["TP"],
                                                                            all_results["mle_ori"]["TN"],
                                                                            all_results["mle_ori"]["FP"],
                                                                            all_results["mle_ori"]["FN"])
                        new_rates_mix_mle_ori[j] = model_lst[2][j].new_rates_easy(all_results["mle_ori"]["A"],
                                                                                  all_results["mle_ori"]["B"])
                    if all_results["lasso"] is not None:
                        fpr_rates_mix_lasso[j] = model_lst[2][j].FPR_easy(all_results["lasso"]["FP"],
                                                                          all_results["lasso"]["TN"])
                        fnr_rates_mix_lasso[j] = model_lst[2][j].FNR_easy(all_results["lasso"]["FN"],
                                                                          all_results["lasso"]["TP"])
                        mcc_rates_mix_lasso[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP"],
                                                                          all_results["lasso"]["TN"],
                                                                          all_results["lasso"]["FP"],
                                                                          all_results["lasso"]["FN"])
                        new_rates_mix_lasso[j] = model_lst[2][j].new_rates_easy(all_results["lasso"]["A"],
                                                                                all_results["lasso"]["B"])
                    if all_results["ada"] is not None:
                        fpr_rates_mix_ada[j] = model_lst[2][j].FPR_easy(all_results["ada"]["FP"],
                                                                        all_results["ada"]["TN"])
                        fnr_rates_mix_ada[j] = model_lst[2][j].FNR_easy(all_results["ada"]["FN"],
                                                                        all_results["ada"]["TP"])
                        mcc_rates_mix_ada[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP"],
                                                                        all_results["ada"]["TN"],
                                                                        all_results["ada"]["FP"],
                                                                        all_results["ada"]["FN"])
                        new_rates_mix_ada[j] = model_lst[2][j].new_rates_easy(all_results["ada"]["A"],
                                                                              all_results["ada"]["B"])

                fpr_arr_mix_mle[i] = np.average(fpr_rates_mix_mle)
                fnr_arr_mix_mle[i] = np.average(fpr_rates_mix_mle)
                mcc_arr_mix_mle[i] = np.average(mcc_rates_mix_mle)
                new_arr_mix_mle[i] = np.nanmean(new_rates_mix_mle)
                fpr_arr_mix_mle_ori[i] = np.average(fpr_rates_mix_mle_ori)
                fnr_arr_mix_mle_ori[i] = np.average(fpr_rates_mix_mle_ori)
                mcc_arr_mix_mle_ori[i] = np.average(mcc_rates_mix_mle_ori)
                new_arr_mix_mle_ori[i] = np.nanmean(new_rates_mix_mle_ori)
                fpr_arr_mix_lasso[i] = np.average(fpr_rates_mix_lasso)
                fnr_arr_mix_lasso[i] = np.average(fnr_rates_mix_lasso)
                mcc_arr_mix_lasso[i] = np.average(mcc_rates_mix_lasso)
                new_arr_mix_lasso[i] = np.nanmean(new_rates_mix_lasso)
                fpr_arr_mix_ada[i] = np.average(fpr_rates_mix_ada)
                fnr_arr_mix_ada[i] = np.average(fnr_rates_mix_ada)
                mcc_arr_mix_ada[i] = np.average(mcc_rates_mix_ada)
                new_arr_mix_ada[i] = np.nanmean(new_rates_mix_ada)
        else:
            K = 0.1
            # T_arr = np.geomspace(20, 900, num=scan_num)
            # hori_arr = np.copy(T_arr) / (2 * np.pi)
            # hori_arr = inf_l * hori_arr
            # hori_arr = hori_arr.astype(int)

            hori_arr = np.array([2, 5, 10, 20, 50, 100, 200])
            T_arr = hori_arr * (2 * np.pi) / inf_l
            scan_num = len(T_arr)

            fpr_arr_2_mle = np.zeros(scan_num)
            fnr_arr_2_mle = np.zeros(scan_num)
            mcc_arr_2_mle = np.zeros(scan_num)
            new_arr_2_mle = np.zeros(scan_num)
            fpr_arr_2_mle_ori = np.zeros(scan_num)
            fnr_arr_2_mle_ori = np.zeros(scan_num)
            mcc_arr_2_mle_ori = np.zeros(scan_num)
            new_arr_2_mle_ori = np.zeros(scan_num)
            fpr_arr_2_lasso = np.zeros(scan_num)
            fnr_arr_2_lasso = np.zeros(scan_num)
            mcc_arr_2_lasso = np.zeros(scan_num)
            new_arr_2_lasso = np.zeros(scan_num)
            fpr_arr_2_ada = np.zeros(scan_num)
            fnr_arr_2_ada = np.zeros(scan_num)
            mcc_arr_2_ada = np.zeros(scan_num)
            new_arr_2_ada = np.zeros(scan_num)

            fpr_arr_3_mle = np.zeros(scan_num)
            fnr_arr_3_mle = np.zeros(scan_num)
            mcc_arr_3_mle = np.zeros(scan_num)
            new_arr_3_mle = np.zeros(scan_num)
            fpr_arr_3_mle_ori = np.zeros(scan_num)
            fnr_arr_3_mle_ori = np.zeros(scan_num)
            mcc_arr_3_mle_ori = np.zeros(scan_num)
            new_arr_3_mle_ori = np.zeros(scan_num)
            fpr_arr_3_lasso = np.zeros(scan_num)
            fnr_arr_3_lasso = np.zeros(scan_num)
            mcc_arr_3_lasso = np.zeros(scan_num)
            new_arr_3_lasso = np.zeros(scan_num)
            fpr_arr_3_ada = np.zeros(scan_num)
            fnr_arr_3_ada = np.zeros(scan_num)
            mcc_arr_3_ada = np.zeros(scan_num)
            new_arr_3_ada = np.zeros(scan_num)

            fpr_arr_mix_mle = np.zeros(scan_num)
            fnr_arr_mix_mle = np.zeros(scan_num)
            mcc_arr_mix_mle = np.zeros(scan_num)
            new_arr_mix_mle = np.zeros(scan_num)
            fpr_arr_mix_mle_ori = np.zeros(scan_num)
            fnr_arr_mix_mle_ori = np.zeros(scan_num)
            mcc_arr_mix_mle_ori = np.zeros(scan_num)
            new_arr_mix_mle_ori = np.zeros(scan_num)
            fpr_arr_mix_lasso = np.zeros(scan_num)
            fnr_arr_mix_lasso = np.zeros(scan_num)
            mcc_arr_mix_lasso = np.zeros(scan_num)
            new_arr_mix_lasso = np.zeros(scan_num)
            fpr_arr_mix_ada = np.zeros(scan_num)
            fnr_arr_mix_ada = np.zeros(scan_num)
            mcc_arr_mix_ada = np.zeros(scan_num)
            new_arr_mix_ada = np.zeros(scan_num)

            for i in range(scan_num):
                if pre_conn_2_lst is None:
                    pre_conn_2_arr = None
                else:
                    pre_conn_2_arr = pre_conn_2_lst[i]

                if pre_conn_3_lst is None:
                    pre_conn_3_arr = None
                else:
                    pre_conn_3_arr = pre_conn_2_lst[i]

                model_lst, real_conn_2_arr, real_conn_3_arr = create_task_Fig3(conn_seed_=conn_seed_,
                                                                               coup_seed_=coup_seed_,
                                                                               noise_arr=noise_arr,
                                                                               natfreq_arr=natfreq_arr,
                                                                               pre_conn_2_arr=pre_conn_2_arr,
                                                                               trial_num=trial_num, K=K, T=T_arr[i],
                                                                               dt=dt, p_=p_,
                                                                               noise_sth=noise_sth, ratio=ratio,
                                                                               st_fr=st_fr, inf_l=inf_l,
                                                                               reduce_seed=reduce_seed,
                                                                               ops_limit=ops_limit,
                                                                               all_connected=all_connected,
                                                                               pre_conn_3_arr=pre_conn_3_arr)
                if np.any(real_conn_2_arr != 0):
                    print("The outcome conn2_arr is", real_conn_2_arr)
                real_2_lst.append(real_conn_2_arr)
                if np.any(real_conn_3_arr != 0):
                    print("The outcome conn3_arr is", real_conn_3_arr)
                real_3_lst.append(real_conn_3_arr)

                fpr_rates_2_mle = np.zeros(trial_num)
                fnr_rates_2_mle = np.zeros(trial_num)
                mcc_rates_2_mle = np.zeros(trial_num)
                new_rates_2_mle = np.zeros(trial_num)
                fpr_rates_2_mle_ori = np.zeros(trial_num)
                fnr_rates_2_mle_ori = np.zeros(trial_num)
                mcc_rates_2_mle_ori = np.zeros(trial_num)
                new_rates_2_mle_ori = np.zeros(trial_num)
                fpr_rates_2_lasso = np.zeros(trial_num)
                fnr_rates_2_lasso = np.zeros(trial_num)
                mcc_rates_2_lasso = np.zeros(trial_num)
                new_rates_2_lasso = np.zeros(trial_num)
                fpr_rates_2_ada = np.zeros(trial_num)
                fnr_rates_2_ada = np.zeros(trial_num)
                mcc_rates_2_ada = np.zeros(trial_num)
                new_rates_2_ada = np.zeros(trial_num)
                for j in range(trial_num):
                    act_mat = model_lst[0][j].run()
                    prepare_all = model_lst[0][j].prepare_diffs(act_mat)
                    if ols:
                        mle_or_ols_results = model_lst[0][j].solve_ols(all_prepared=prepare_all)
                    else:
                        mle_or_ols_results = None
                    if ada:
                        ada_results = model_lst[0][j].solve_ada_lasso(all_prepared=prepare_all,
                                                                      mle_or_ols_results=mle_or_ols_results)
                    else:
                        ada_results = None
                    if lasso:
                        lasso_results = model_lst[0][j].solve_lasso(all_prepared=prepare_all)
                    else:
                        lasso_results = None
                    all_results = model_lst[0][j].conn_criteria_base(mle_or_ols_results=mle_or_ols_results,
                                                                     lasso_results=lasso_results,
                                                                     ada_results=ada_results,
                                                                     mle_ori=mle_ori, mle_threshold=mle_threshold)
                    if all_results["mle"] is not None:
                        fpr_rates_2_mle[j] = model_lst[0][j].FPR_easy(all_results["mle"]["FP"],
                                                                      all_results["mle"]["TN"])
                        fnr_rates_2_mle[j] = model_lst[0][j].FNR_easy(all_results["mle"]["FN"],
                                                                      all_results["mle"]["TP"])
                        mcc_rates_2_mle[j] = model_lst[0][j].MCC_easy(all_results["mle"]["TP"],
                                                                      all_results["mle"]["TN"],
                                                                      all_results["mle"]["FP"],
                                                                      all_results["mle"]["FN"])
                        new_rates_2_mle[j] = model_lst[0][j].new_rates_easy(all_results["mle"]["A"],
                                                                            all_results["mle"]["B"])
                    if all_results["mle_ori"] is not None:
                        fpr_rates_2_mle_ori[j] = model_lst[0][j].FPR_easy(all_results["mle_ori"]["FP"],
                                                                          all_results["mle_ori"]["TN"])
                        fnr_rates_2_mle_ori[j] = model_lst[0][j].FNR_easy(all_results["mle_ori"]["FN"],
                                                                          all_results["mle_ori"]["TP"])
                        mcc_rates_2_mle_ori[j] = model_lst[0][j].MCC_easy(all_results["mle_ori"]["TP"],
                                                                          all_results["mle_ori"]["TN"],
                                                                          all_results["mle_ori"]["FP"],
                                                                          all_results["mle_ori"]["FN"])
                        new_rates_2_mle_ori[j] = model_lst[0][j].new_rates_easy(all_results["mle_ori"]["A"],
                                                                                all_results["mle_ori"]["B"])
                    if all_results["lasso"] is not None:
                        fpr_rates_2_lasso[j] = model_lst[0][j].FPR_easy(all_results["lasso"]["FP"],
                                                                        all_results["lasso"]["TN"])
                        fnr_rates_2_lasso[j] = model_lst[0][j].FNR_easy(all_results["lasso"]["FN"],
                                                                        all_results["lasso"]["TP"])
                        mcc_rates_2_lasso[j] = model_lst[0][j].MCC_easy(all_results["lasso"]["TP"],
                                                                        all_results["lasso"]["TN"],
                                                                        all_results["lasso"]["FP"],
                                                                        all_results["lasso"]["FN"])
                        new_rates_2_lasso[j] = model_lst[0][j].new_rates_easy(all_results["lasso"]["A"],
                                                                              all_results["lasso"]["B"])
                    if all_results["ada"] is not None:
                        fpr_rates_2_ada[j] = model_lst[0][j].FPR_easy(all_results["ada"]["FP"],
                                                                      all_results["ada"]["TN"])
                        fnr_rates_2_ada[j] = model_lst[0][j].FNR_easy(all_results["ada"]["FN"],
                                                                      all_results["ada"]["TP"])
                        mcc_rates_2_ada[j] = model_lst[0][j].MCC_easy(all_results["ada"]["TP"],
                                                                      all_results["ada"]["TN"],
                                                                      all_results["ada"]["FP"],
                                                                      all_results["ada"]["FN"])
                        new_rates_2_ada[j] = model_lst[0][j].new_rates_easy(all_results["ada"]["A"],
                                                                            all_results["ada"]["B"])

                fpr_arr_2_mle[i] = np.average(fpr_rates_2_mle)
                fnr_arr_2_mle[i] = np.average(fpr_rates_2_mle)
                mcc_arr_2_mle[i] = np.average(mcc_rates_2_mle)
                new_arr_2_mle[i] = np.nanmean(new_rates_2_mle)
                fpr_arr_2_mle_ori[i] = np.average(fpr_rates_2_mle_ori)
                fnr_arr_2_mle_ori[i] = np.average(fpr_rates_2_mle_ori)
                mcc_arr_2_mle_ori[i] = np.average(mcc_rates_2_mle_ori)
                new_arr_2_mle_ori[i] = np.nanmean(new_rates_2_mle_ori)
                fpr_arr_2_lasso[i] = np.average(fpr_rates_2_lasso)
                fnr_arr_2_lasso[i] = np.average(fnr_rates_2_lasso)
                mcc_arr_2_lasso[i] = np.average(mcc_rates_2_lasso)
                new_arr_2_lasso[i] = np.nanmean(new_rates_2_lasso)
                fpr_arr_2_ada[i] = np.average(fpr_rates_2_ada)
                fnr_arr_2_ada[i] = np.average(fnr_rates_2_ada)
                mcc_arr_2_ada[i] = np.average(mcc_rates_2_ada)
                new_arr_2_ada[i] = np.nanmean(new_rates_2_ada)

                fpr_rates_3_mle = np.zeros(trial_num)
                fnr_rates_3_mle = np.zeros(trial_num)
                mcc_rates_3_mle = np.zeros(trial_num)
                new_rates_3_mle = np.zeros(trial_num)
                fpr_rates_3_mle_ori = np.zeros(trial_num)
                fnr_rates_3_mle_ori = np.zeros(trial_num)
                mcc_rates_3_mle_ori = np.zeros(trial_num)
                new_rates_3_mle_ori = np.zeros(trial_num)
                fpr_rates_3_lasso = np.zeros(trial_num)
                fnr_rates_3_lasso = np.zeros(trial_num)
                mcc_rates_3_lasso = np.zeros(trial_num)
                new_rates_3_lasso = np.zeros(trial_num)
                fpr_rates_3_ada = np.zeros(trial_num)
                fnr_rates_3_ada = np.zeros(trial_num)
                mcc_rates_3_ada = np.zeros(trial_num)
                new_rates_3_ada = np.zeros(trial_num)
                for j in range(trial_num):
                    act_mat = model_lst[1][j].run()
                    prepare_all = model_lst[1][j].prepare_diffs(act_mat)
                    if ols:
                        mle_or_ols_results = model_lst[1][j].solve_ols(all_prepared=prepare_all)
                    else:
                        mle_or_ols_results = None
                    if ada:
                        ada_results = model_lst[1][j].solve_ada_lasso(all_prepared=prepare_all,
                                                                      mle_or_ols_results=mle_or_ols_results)
                    else:
                        ada_results = None
                    if lasso:
                        lasso_results = model_lst[1][j].solve_lasso(all_prepared=prepare_all)
                    else:
                        lasso_results = None
                    all_results = model_lst[1][j].conn_criteria_base(mle_or_ols_results=mle_or_ols_results,
                                                                     lasso_results=lasso_results,
                                                                     ada_results=ada_results,
                                                                     mle_ori=mle_ori, mle_threshold=mle_threshold)
                    if all_results["mle"] is not None:
                        fpr_rates_3_mle[j] = model_lst[1][j].FPR_easy(all_results["mle"]["FP"],
                                                                      all_results["mle"]["TN"])
                        fnr_rates_3_mle[j] = model_lst[1][j].FNR_easy(all_results["mle"]["FN"],
                                                                      all_results["mle"]["TP"])
                        mcc_rates_3_mle[j] = model_lst[1][j].MCC_easy(all_results["mle"]["TP"],
                                                                      all_results["mle"]["TN"],
                                                                      all_results["mle"]["FP"],
                                                                      all_results["mle"]["FN"])
                        new_rates_3_mle[j] = model_lst[1][j].new_rates_easy(all_results["mle"]["A"],
                                                                            all_results["mle"]["B"])
                    if all_results["mle_ori"] is not None:
                        fpr_rates_3_mle_ori[j] = model_lst[1][j].FPR_easy(all_results["mle_ori"]["FP"],
                                                                          all_results["mle_ori"]["TN"])
                        fnr_rates_3_mle_ori[j] = model_lst[1][j].FNR_easy(all_results["mle_ori"]["FN"],
                                                                          all_results["mle_ori"]["TP"])
                        mcc_rates_3_mle_ori[j] = model_lst[1][j].MCC_easy(all_results["mle_ori"]["TP"],
                                                                          all_results["mle_ori"]["TN"],
                                                                          all_results["mle_ori"]["FP"],
                                                                          all_results["mle_ori"]["FN"])
                        new_rates_3_mle_ori[j] = model_lst[1][j].new_rates_easy(all_results["mle_ori"]["A"],
                                                                                all_results["mle_ori"]["B"])
                    if all_results["lasso"] is not None:
                        fpr_rates_3_lasso[j] = model_lst[1][j].FPR_easy(all_results["lasso"]["FP"],
                                                                        all_results["lasso"]["TN"])
                        fnr_rates_3_lasso[j] = model_lst[1][j].FNR_easy(all_results["lasso"]["FN"],
                                                                        all_results["lasso"]["TP"])
                        mcc_rates_3_lasso[j] = model_lst[1][j].MCC_easy(all_results["lasso"]["TP"],
                                                                        all_results["lasso"]["TN"],
                                                                        all_results["lasso"]["FP"],
                                                                        all_results["lasso"]["FN"])
                        new_rates_3_lasso[j] = model_lst[1][j].new_rates_easy(all_results["lasso"]["A"],
                                                                              all_results["lasso"]["B"])
                    if all_results["ada"] is not None:
                        fpr_rates_3_ada[j] = model_lst[1][j].FPR_easy(all_results["ada"]["FP"],
                                                                      all_results["ada"]["TN"])
                        fnr_rates_3_ada[j] = model_lst[1][j].FNR_easy(all_results["ada"]["FN"],
                                                                      all_results["ada"]["TP"])
                        mcc_rates_3_ada[j] = model_lst[1][j].MCC_easy(all_results["ada"]["TP"],
                                                                      all_results["ada"]["TN"],
                                                                      all_results["ada"]["FP"],
                                                                      all_results["ada"]["FN"])
                        new_rates_3_ada[j] = model_lst[1][j].new_rates_easy(all_results["ada"]["A"],
                                                                            all_results["ada"]["B"])

                fpr_arr_3_mle[i] = np.average(fpr_rates_3_mle)
                fnr_arr_3_mle[i] = np.average(fpr_rates_3_mle)
                mcc_arr_3_mle[i] = np.average(mcc_rates_3_mle)
                new_arr_3_mle[i] = np.nanmean(new_rates_3_mle)
                fpr_arr_3_mle_ori[i] = np.average(fpr_rates_3_mle_ori)
                fnr_arr_3_mle_ori[i] = np.average(fpr_rates_3_mle_ori)
                mcc_arr_3_mle_ori[i] = np.average(mcc_rates_3_mle_ori)
                new_arr_3_mle_ori[i] = np.nanmean(new_rates_3_mle_ori)
                fpr_arr_3_lasso[i] = np.average(fpr_rates_3_lasso)
                fnr_arr_3_lasso[i] = np.average(fnr_rates_3_lasso)
                mcc_arr_3_lasso[i] = np.average(mcc_rates_3_lasso)
                new_arr_3_lasso[i] = np.nanmean(new_rates_3_lasso)
                fpr_arr_3_ada[i] = np.average(fpr_rates_3_ada)
                fnr_arr_3_ada[i] = np.average(fnr_rates_3_ada)
                mcc_arr_3_ada[i] = np.average(mcc_rates_3_ada)
                new_arr_3_ada[i] = np.nanmean(new_rates_3_ada)

                fpr_rates_mix_mle = np.zeros(trial_num)
                fnr_rates_mix_mle = np.zeros(trial_num)
                mcc_rates_mix_mle = np.zeros(trial_num)
                new_rates_mix_mle = np.zeros(trial_num)
                fpr_rates_mix_mle_ori = np.zeros(trial_num)
                fnr_rates_mix_mle_ori = np.zeros(trial_num)
                mcc_rates_mix_mle_ori = np.zeros(trial_num)
                new_rates_mix_mle_ori = np.zeros(trial_num)
                fpr_rates_mix_lasso = np.zeros(trial_num)
                fnr_rates_mix_lasso = np.zeros(trial_num)
                mcc_rates_mix_lasso = np.zeros(trial_num)
                new_rates_mix_lasso = np.zeros(trial_num)
                fpr_rates_mix_ada = np.zeros(trial_num)
                fnr_rates_mix_ada = np.zeros(trial_num)
                mcc_rates_mix_ada = np.zeros(trial_num)
                new_rates_mix_ada = np.zeros(trial_num)
                for j in range(trial_num):
                    act_mat = model_lst[2][j].run()
                    prepare_all = model_lst[2][j].prepare_diffs(act_mat)
                    if ols:
                        mle_or_ols_results = model_lst[2][j].solve_ols(all_prepared=prepare_all)
                    else:
                        mle_or_ols_results = None
                    if ada:
                        ada_results = model_lst[2][j].solve_ada_lasso(all_prepared=prepare_all,
                                                                      mle_or_ols_results=mle_or_ols_results)
                    else:
                        ada_results = None
                    if lasso:
                        lasso_results = model_lst[2][j].solve_lasso(all_prepared=prepare_all)
                    else:
                        lasso_results = None
                    all_results = model_lst[2][j].conn_criteria_base(mle_or_ols_results=mle_or_ols_results,
                                                                     lasso_results=lasso_results,
                                                                     ada_results=ada_results,
                                                                     mle_ori=mle_ori, mle_threshold=mle_threshold)
                    if all_results["mle"] is not None:
                        fpr_rates_mix_mle[j] = model_lst[2][j].FPR_easy(all_results["mle"]["FP"],
                                                                        all_results["mle"]["TN"])
                        fnr_rates_mix_mle[j] = model_lst[2][j].FNR_easy(all_results["mle"]["FN"],
                                                                        all_results["mle"]["TP"])
                        mcc_rates_mix_mle[j] = model_lst[2][j].MCC_easy(all_results["mle"]["TP"],
                                                                        all_results["mle"]["TN"],
                                                                        all_results["mle"]["FP"],
                                                                        all_results["mle"]["FN"])
                        new_rates_mix_mle[j] = model_lst[2][j].new_rates_easy(all_results["mle"]["A"],
                                                                              all_results["mle"]["B"])
                    if all_results["mle_ori"] is not None:
                        fpr_rates_mix_mle_ori[j] = model_lst[2][j].FPR_easy(all_results["mle_ori"]["FP"],
                                                                            all_results["mle_ori"]["TN"])
                        fnr_rates_mix_mle_ori[j] = model_lst[2][j].FNR_easy(all_results["mle_ori"]["FN"],
                                                                            all_results["mle_ori"]["TP"])
                        mcc_rates_mix_mle_ori[j] = model_lst[2][j].MCC_easy(all_results["mle_ori"]["TP"],
                                                                            all_results["mle_ori"]["TN"],
                                                                            all_results["mle_ori"]["FP"],
                                                                            all_results["mle_ori"]["FN"])
                        new_rates_mix_mle_ori[j] = model_lst[2][j].new_rates_easy(all_results["mle_ori"]["A"],
                                                                                  all_results["mle_ori"]["B"])
                    if all_results["lasso"] is not None:
                        fpr_rates_mix_lasso[j] = model_lst[2][j].FPR_easy(all_results["lasso"]["FP"],
                                                                          all_results["lasso"]["TN"])
                        fnr_rates_mix_lasso[j] = model_lst[2][j].FNR_easy(all_results["lasso"]["FN"],
                                                                          all_results["lasso"]["TP"])
                        mcc_rates_mix_lasso[j] = model_lst[2][j].MCC_easy(all_results["lasso"]["TP"],
                                                                          all_results["lasso"]["TN"],
                                                                          all_results["lasso"]["FP"],
                                                                          all_results["lasso"]["FN"])
                        new_rates_mix_lasso[j] = model_lst[2][j].new_rates_easy(all_results["lasso"]["A"],
                                                                                all_results["lasso"]["B"])
                    if all_results["ada"] is not None:
                        fpr_rates_mix_ada[j] = model_lst[2][j].FPR_easy(all_results["ada"]["FP"],
                                                                        all_results["ada"]["TN"])
                        fnr_rates_mix_ada[j] = model_lst[2][j].FNR_easy(all_results["ada"]["FN"],
                                                                        all_results["ada"]["TP"])
                        mcc_rates_mix_ada[j] = model_lst[2][j].MCC_easy(all_results["ada"]["TP"],
                                                                        all_results["ada"]["TN"],
                                                                        all_results["ada"]["FP"],
                                                                        all_results["ada"]["FN"])
                        new_rates_mix_ada[j] = model_lst[2][j].new_rates_easy(all_results["ada"]["A"],
                                                                              all_results["ada"]["B"])

                fpr_arr_mix_mle[i] = np.average(fpr_rates_mix_mle)
                fnr_arr_mix_mle[i] = np.average(fpr_rates_mix_mle)
                mcc_arr_mix_mle[i] = np.average(mcc_rates_mix_mle)
                new_arr_mix_mle[i] = np.nanmean(new_rates_mix_mle)
                fpr_arr_mix_mle_ori[i] = np.average(fpr_rates_mix_mle_ori)
                fnr_arr_mix_mle_ori[i] = np.average(fpr_rates_mix_mle_ori)
                mcc_arr_mix_mle_ori[i] = np.average(mcc_rates_mix_mle_ori)
                new_arr_mix_mle_ori[i] = np.nanmean(new_rates_mix_mle_ori)
                fpr_arr_mix_lasso[i] = np.average(fpr_rates_mix_lasso)
                fnr_arr_mix_lasso[i] = np.average(fnr_rates_mix_lasso)
                mcc_arr_mix_lasso[i] = np.average(mcc_rates_mix_lasso)
                new_arr_mix_lasso[i] = np.nanmean(new_rates_mix_lasso)
                fpr_arr_mix_ada[i] = np.average(fpr_rates_mix_ada)
                fnr_arr_mix_ada[i] = np.average(fnr_rates_mix_ada)
                mcc_arr_mix_ada[i] = np.average(mcc_rates_mix_ada)
                new_arr_mix_ada[i] = np.nanmean(new_rates_mix_ada)

        output = {"x": hori_arr,
                  "ada": [new_arr_2_ada, new_arr_3_ada, new_arr_mix_ada,
                          fpr_arr_2_ada, fpr_arr_3_ada, fpr_arr_mix_ada,
                          fnr_arr_2_ada, fnr_arr_3_ada, fnr_arr_mix_ada,
                          mcc_arr_2_ada, mcc_arr_3_ada, mcc_arr_mix_ada],
                  "lasso": [new_arr_2_lasso, new_arr_3_lasso, new_arr_mix_lasso,
                            fpr_arr_2_lasso, fpr_arr_3_lasso, fpr_arr_mix_lasso,
                            fnr_arr_2_lasso, fnr_arr_3_lasso, fnr_arr_mix_lasso,
                            mcc_arr_2_lasso, mcc_arr_3_lasso, mcc_arr_mix_lasso],
                  "mle": [new_arr_2_mle, new_arr_3_mle, new_arr_mix_mle,
                          fpr_arr_2_mle, fpr_arr_3_mle, fpr_arr_mix_mle,
                          fnr_arr_2_mle, fnr_arr_3_mle, fnr_arr_mix_mle,
                          mcc_arr_2_mle, mcc_arr_3_mle, mcc_arr_mix_mle],
                  "mle_ori": [new_arr_2_mle_ori, new_arr_3_mle_ori, new_arr_mix_mle_ori,
                              fpr_arr_2_mle_ori, fpr_arr_3_mle_ori, fpr_arr_mix_mle_ori,
                              fnr_arr_2_mle_ori, fnr_arr_3_mle_ori, fnr_arr_mix_mle_ori,
                              mcc_arr_2_mle_ori, mcc_arr_3_mle_ori, mcc_arr_mix_mle_ori]
                  }

        if draw == 1:
            drawing_1(output, p_, paths, K_or_T)
        elif draw == 2:
            outputs.append(output)

        np.save(f"{paths}new_two_seeds", real_2_lst)
        np.save(f"{paths}new_three_seeds", real_3_lst)

    if draw == 2:
        drawing_mcc_only(outputs, p_lst_, file_path_p_lst, file_path_base, sharex=True, K_or_T=K_or_T, ada=ada,
                         lasso=lasso, mle=ols, mle_ori=mle_ori)


def drawing_1(output, p_, paths, K_or_T=1):
    hori_arr = output["x"]

    new_arr_2_ada = output["ada"][0]
    new_arr_3_ada = output["ada"][1]
    new_arr_mix_ada = output["ada"][2]
    fpr_arr_2_ada = output["ada"][3]
    fpr_arr_3_ada = output["ada"][4]
    fpr_arr_mix_ada = output["ada"][5]
    fnr_arr_2_ada = output["ada"][6]
    fnr_arr_3_ada = output["ada"][7]
    fnr_arr_mix_ada = output["ada"][8]
    mcc_arr_2_ada = output["ada"][9]
    mcc_arr_3_ada = output["ada"][10]
    mcc_arr_mix_ada = output["ada"][11]

    new_arr_2_lasso = output["lasso"][0]
    new_arr_3_lasso = output["lasso"][1]
    new_arr_mix_lasso = output["lasso"][2]
    fpr_arr_2_lasso = output["lasso"][3]
    fpr_arr_3_lasso = output["lasso"][4]
    fpr_arr_mix_lasso = output["lasso"][5]
    fnr_arr_2_lasso = output["lasso"][6]
    fnr_arr_3_lasso = output["lasso"][7]
    fnr_arr_mix_lasso = output["lasso"][8]
    mcc_arr_2_lasso = output["lasso"][9]
    mcc_arr_3_lasso = output["lasso"][10]
    mcc_arr_mix_lasso = output["lasso"][11]

    new_arr_2_mle = output["mle"][0]
    new_arr_3_mle = output["mle"][1]
    new_arr_mix_mle = output["mle"][2]
    fpr_arr_2_mle = output["mle"][3]
    fpr_arr_3_mle = output["mle"][4]
    fpr_arr_mix_mle = output["mle"][5]
    fnr_arr_2_mle = output["mle"][6]
    fnr_arr_3_mle = output["mle"][7]
    fnr_arr_mix_mle = output["mle"][8]
    mcc_arr_2_mle = output["mle"][9]
    mcc_arr_3_mle = output["mle"][10]
    mcc_arr_mix_mle = output["mle"][11]

    fig1 = plt.figure(figsize=(20, 10), dpi=100)
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(20, 10), dpi=100)
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(figsize=(20, 10), dpi=100)
    ax3 = fig3.add_subplot(111)
    fig5 = plt.figure(figsize=(20, 10), dpi=100)
    ax5 = fig5.add_subplot(111)
    fig6 = plt.figure(figsize=(20, 10), dpi=100)
    ax6 = fig6.add_subplot(111)
    fig7 = plt.figure(figsize=(20, 10), dpi=100)
    ax7 = fig7.add_subplot(111)
    fig8 = plt.figure(figsize=(20, 10), dpi=100)
    ax8 = fig8.add_subplot(111)
    fig9 = plt.figure(figsize=(20, 10), dpi=100)
    ax9 = fig9.add_subplot(111)
    fig10 = plt.figure(figsize=(20, 10), dpi=100)
    ax10 = fig10.add_subplot(111)
    fig11 = plt.figure(figsize=(20, 10), dpi=100)
    ax11 = fig11.add_subplot(111)
    fig12 = plt.figure(figsize=(20, 10), dpi=100)
    ax12 = fig12.add_subplot(111)
    fig13 = plt.figure(figsize=(20, 10), dpi=100)
    ax13 = fig13.add_subplot(111)

    ax1.plot(hori_arr, new_arr_2_ada, label="Pairwise, A. LASSO")
    ax2.plot(hori_arr, new_arr_3_ada, label=f"3-int, A. LASSO")
    ax3.plot(hori_arr, new_arr_mix_ada, label=f"Mixture, A. LASSO")
    ax5.plot(hori_arr, fpr_arr_2_ada, label="Pairwise FPR, A. LASSO")
    ax6.plot(hori_arr, fpr_arr_3_ada, label="3-int FPR, A. LASSO")
    ax7.plot(hori_arr, fpr_arr_mix_ada, label="Mix FPR, A. LASSO")
    ax8.plot(hori_arr, fnr_arr_2_ada, label="Pairwise FNR, A. LASSO")
    ax9.plot(hori_arr, fnr_arr_3_ada, label="3-int FNR, A. LASSO")
    ax10.plot(hori_arr, fnr_arr_mix_ada, label="Mix FNR, A. LASSO")
    ax11.plot(hori_arr, mcc_arr_2_ada, label="Pairwise MCC, A. LASSO")
    ax12.plot(hori_arr, mcc_arr_3_ada, label="3-int MCC, A. LASSO")
    ax13.plot(hori_arr, mcc_arr_mix_ada, label="Mix MCC, A. LASSO")

    ax1.plot(hori_arr, new_arr_2_lasso, label="Pairwise, LASSO", linewidth=5, alpha=0.6)
    ax2.plot(hori_arr, new_arr_3_lasso, label=f"3-int, LASSO", linewidth=5, alpha=0.6)
    ax3.plot(hori_arr, new_arr_mix_lasso, label=f"Mixture, LASSO", linewidth=5, alpha=0.6)
    ax5.plot(hori_arr, fpr_arr_2_lasso, label="Pairwise FPR, LASSO", linewidth=5, alpha=0.6)
    ax6.plot(hori_arr, fpr_arr_3_lasso, label="3-int FPR, LASSO", linewidth=5, alpha=0.6)
    ax7.plot(hori_arr, fpr_arr_mix_lasso, label="Mix FPR, LASSO", linewidth=5, alpha=0.6)
    ax8.plot(hori_arr, fnr_arr_2_lasso, label="Pairwise FNR, LASSO", linewidth=5, alpha=0.6)
    ax9.plot(hori_arr, fnr_arr_3_lasso, label="3-int FNR, LASSO", linewidth=5, alpha=0.6)
    ax10.plot(hori_arr, fnr_arr_mix_lasso, label="Mix FNR, LASSO", linewidth=5, alpha=0.6)
    ax11.plot(hori_arr, mcc_arr_2_lasso, label="Pairwise MCC, LASSO", linewidth=5, alpha=0.6)
    ax12.plot(hori_arr, mcc_arr_3_lasso, label="3-int MCC, LASSO", linewidth=5, alpha=0.6)
    ax13.plot(hori_arr, mcc_arr_mix_lasso, label="Mix MCC, LASSO", linewidth=5, alpha=0.6)

    ax1.plot(hori_arr, new_arr_2_mle, label="Pairwise, OLS")
    ax2.plot(hori_arr, new_arr_3_mle, label=f"3-int, OLS")
    ax3.plot(hori_arr, new_arr_mix_mle, label=f"Mixture, OLS")
    ax5.plot(hori_arr, fpr_arr_2_mle, label="Pairwise FPR, OLS")
    ax6.plot(hori_arr, fpr_arr_3_mle, label="3-int FPR, OLS")
    ax7.plot(hori_arr, fpr_arr_mix_mle, label="Mix FPR, OLS")
    ax8.plot(hori_arr, fnr_arr_2_mle, label="Pairwise FNR, OLS")
    ax9.plot(hori_arr, fnr_arr_3_mle, label="3-int FNR, OLS")
    ax10.plot(hori_arr, fnr_arr_mix_mle, label="Mix FNR, OLS")
    ax11.plot(hori_arr, mcc_arr_2_mle, label="Pairwise MCC, OLS")
    ax12.plot(hori_arr, mcc_arr_3_mle, label="3-int MCC, OLS")
    ax13.plot(hori_arr, mcc_arr_mix_mle, label="Mix MCC, OLS")

    ax3.axhline(0.5, linestyle="--", label="value 0.5")

    plt.rcParams["font.size"] = 36  # フォントを大きく
    plt.rcParams['font.family'] = 'Arial'
    params = {'legend.fontsize': 32,
              'legend.handlelength': 1}
    plt.rcParams.update(params)
    if K_or_T == 1:
        xlabel = 'Coupling strength'
    else:
        xlabel = 'Cycle number'
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax3.set_xscale("log")
        ax5.set_xscale("log")
        ax6.set_xscale("log")
        ax7.set_xscale("log")
        ax8.set_xscale("log")
        ax9.set_xscale("log")
        ax10.set_xscale("log")
        ax11.set_xscale("log")
        ax12.set_xscale("log")
        ax13.set_xscale("log")
    ax1.set(xlabel=xlabel)
    ax1.set_ylabel('A/(A+B)', rotation=90, fontsize=29)
    ax1.set_xticks(hori_arr)
    ax1.tick_params(axis='x', labelsize=25)
    ax1.tick_params(axis='y', labelsize=25)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set(xlabel=xlabel)
    ax2.set_ylabel('A/(A+B)', rotation=90, fontsize=29)
    ax2.set_xticks(hori_arr)
    ax2.tick_params(axis='x', labelsize=25)
    ax2.tick_params(axis='y', labelsize=25)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.set(xlabel=xlabel)
    ax3.set_ylabel('A/(A+B)', rotation=90, fontsize=29)
    ax3.set_xticks(hori_arr)
    ax3.tick_params(axis='x', labelsize=25)
    ax3.tick_params(axis='y', labelsize=25)
    ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax5.set(xlabel=xlabel)
    ax5.set_ylabel('FPR', rotation=90, fontsize=29)
    ax5.set_xticks(hori_arr)
    ax5.tick_params(axis='x', labelsize=25)
    ax5.tick_params(axis='y', labelsize=25)
    ax5.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax6.set(xlabel=xlabel)
    ax6.set_ylabel('FPR', rotation=90, fontsize=29)
    ax6.set_xticks(hori_arr)
    ax6.tick_params(axis='x', labelsize=25)
    ax6.tick_params(axis='y', labelsize=25)
    ax6.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax7.set(xlabel=xlabel)
    ax7.set_ylabel('FPR', rotation=90, fontsize=29)
    ax7.set_xticks(hori_arr)
    ax7.tick_params(axis='x', labelsize=25)
    ax7.tick_params(axis='y', labelsize=25)
    ax7.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax8.set(xlabel=xlabel)
    ax8.set_ylabel('FNR', rotation=90, fontsize=29)
    ax8.set_xticks(hori_arr)
    ax8.tick_params(axis='x', labelsize=25)
    ax8.tick_params(axis='y', labelsize=25)
    ax8.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax9.set(xlabel=xlabel)
    ax9.set_ylabel('FNR', rotation=90, fontsize=29)
    ax9.set_xticks(hori_arr)
    ax9.tick_params(axis='x', labelsize=25)
    ax9.tick_params(axis='y', labelsize=25)
    ax9.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax10.set(xlabel=xlabel)
    ax10.set_ylabel('FNR', rotation=90, fontsize=29)
    ax10.set_xticks(hori_arr)
    ax10.tick_params(axis='x', labelsize=25)
    ax10.tick_params(axis='y', labelsize=25)
    ax10.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax11.set(xlabel=xlabel)
    ax11.set_ylabel('MCC', rotation=90, fontsize=29)
    ax11.set_xticks(hori_arr)
    ax11.tick_params(axis='x', labelsize=25)
    ax11.tick_params(axis='y', labelsize=25)
    ax11.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax12.set(xlabel=xlabel)
    ax12.set_ylabel('MCC', rotation=90, fontsize=29)
    ax12.set_xticks(hori_arr)
    ax12.tick_params(axis='x', labelsize=25)
    ax12.tick_params(axis='y', labelsize=25)
    ax12.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax13.set(xlabel=xlabel)
    ax13.set_ylabel('MCC', rotation=90, fontsize=29)
    ax13.set_xticks(hori_arr)
    ax13.tick_params(axis='x', labelsize=25)
    ax13.tick_params(axis='y', labelsize=25)
    ax13.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.yaxis.set_label_coords(-0.08, 0.5)
    # ax.yaxis.set_major_formatter(formatter)
    title_1 = f"Interaction type inference $p={p_}$"
    title_2 = f"Interaction type inference $p={p_}$"
    title_3 = f"Interaction type inference $p={p_}$"
    title_5 = f"False Positive Rate - Pairwise $p={p_}$"
    title_6 = f"False Positive Rate - 3-int $p={p_}$"
    title_7 = f"False Positive Rate - Mix $p={p_}$"
    title_8 = f"False Negative Rate - Pairwise $p={p_}$"
    title_9 = f"False Negative Rate - 3-int $p={p_}$"
    title_10 = f"False Negative Rate - Mix $p={p_}$"
    title_11 = f"Matthews Correlation Coefficient - Pairwise $p={p_}$"
    title_12 = f"Matthews Correlation Coefficient - 3-int $p={p_}$"
    title_13 = f"Matthews Correlation Coefficient - Mix $p={p_}$"
    ax1.set_title(title_1, fontsize=29)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax2.set_title(title_2, fontsize=29)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax3.set_title(title_3, fontsize=29)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend()
    ax5.set_title(title_5, fontsize=29)
    ax5.set_ylim(-0.05, 1.05)
    ax5.legend()
    ax6.set_title(title_6, fontsize=29)
    ax6.set_ylim(-0.05, 1.05)
    ax6.legend()
    ax7.set_title(title_7, fontsize=29)
    ax7.set_ylim(-0.05, 1.05)
    ax7.legend()
    ax8.set_title(title_8, fontsize=29)
    ax8.set_ylim(-0.05, 1.05)
    ax8.legend()
    ax9.set_title(title_9, fontsize=29)
    ax9.set_ylim(-0.05, 1.05)
    ax9.legend()
    ax10.set_title(title_10, fontsize=29)
    ax10.set_ylim(-0.05, 1.05)
    ax10.legend()
    ax11.set_title(title_11, fontsize=29)
    ax11.set_ylim(-1.05, 1.05)
    ax11.legend()
    ax12.set_title(title_12, fontsize=29)
    ax12.set_ylim(-1.05, 1.05)
    ax12.legend()
    ax13.set_title(title_13, fontsize=29)
    ax13.set_ylim(-1.05, 1.05)
    ax13.legend()

    path1 = paths + "pairwise_rate.jpg"
    path2 = paths + "3-inteaction_rate.jpg"
    path3 = paths + "mixture_rate.jpg"
    path5 = paths + "FPR_2.jpg"
    path6 = paths + "FPR_3.jpg"
    path7 = paths + "FPR_mix.jpg"
    path8 = paths + "FNR_2.jpg"
    path9 = paths + "FNR_3.jpg"
    path10 = paths + "FNR_mix.jpg"
    path11 = paths + "MCC_2.jpg"
    path12 = paths + "MCC_3.jpg"
    path13 = paths + "MCC_mix.jpg"
    fig1.savefig(path1)
    fig2.savefig(path2)
    fig3.savefig(path3)
    fig5.savefig(path5)
    fig6.savefig(path6)
    fig7.savefig(path7)
    fig8.savefig(path8)
    fig9.savefig(path9)
    fig10.savefig(path10)
    fig11.savefig(path11)
    fig12.savefig(path12)
    fig13.savefig(path13)

    fpr_txt_2_pth = paths + "FPR_pairwise.csv"
    fpr_txt_3_pth = paths + "FPR_3-interaction.csv"
    fpr_txt_mix_pth = paths + "FPR_mixture.csv"
    fnr_txt_2_pth = paths + "FNR_pairwise.csv"
    fnr_txt_3_pth = paths + "FNR_3-interaction.csv"
    fnr_txt_mix_pth = paths + "FNR_mixture.csv"
    mcc_txt_2_pth = paths + "MCC_pairwise.csv"
    mcc_txt_3_pth = paths + "MCC_3-interaction.csv"
    mcc_txt_mix_pth = paths + "MCC_mixture.csv"

    with open(fpr_txt_2_pth, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if K_or_T == 1:
            writer.writerow(["Coupling Strength"] + hori_arr.tolist())
        else:
            writer.writerow(["# of cycles"] + hori_arr.tolist())
        writer.writerow(["Adaptive LASSO"] + fpr_arr_2_ada.tolist())
        writer.writerow(["LASSO"] + fpr_arr_2_lasso.tolist())
        writer.writerow(["OLS"] + fpr_arr_2_mle.tolist())
        f.close()

    with open(fpr_txt_3_pth, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if K_or_T == 1:
            writer.writerow(["Coupling Strength"] + hori_arr.tolist())
        else:
            writer.writerow(["# of cycles"] + hori_arr.tolist())
        writer.writerow(["Adaptive LASSO"] + fpr_arr_3_ada.tolist())
        writer.writerow(["LASSO"] + fpr_arr_3_lasso.tolist())
        writer.writerow(["OLS"] + fpr_arr_3_mle.tolist())
        f.close()

    with open(fpr_txt_mix_pth, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if K_or_T == 1:
            writer.writerow(["Coupling Strength"] + hori_arr.tolist())
        else:
            writer.writerow(["# of cycles"] + hori_arr.tolist())
        writer.writerow(["Adaptive LASSO"] + fpr_arr_mix_ada.tolist())
        writer.writerow(["LASSO"] + fpr_arr_mix_lasso.tolist())
        writer.writerow(["OLS"] + fpr_arr_mix_mle.tolist())
        f.close()

    with open(fnr_txt_2_pth, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if K_or_T == 1:
            writer.writerow(["Coupling Strength"] + hori_arr.tolist())
        else:
            writer.writerow(["# of cycles"] + hori_arr.tolist())
        writer.writerow(["Adaptive LASSO"] + fnr_arr_2_ada.tolist())
        writer.writerow(["LASSO"] + fnr_arr_2_lasso.tolist())
        writer.writerow(["OLS"] + fnr_arr_2_mle.tolist())
        f.close()

    with open(fnr_txt_3_pth, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if K_or_T == 1:
            writer.writerow(["Coupling Strength"] + hori_arr.tolist())
        else:
            writer.writerow(["# of cycles"] + hori_arr.tolist())
        writer.writerow(["Adaptive LASSO"] + fnr_arr_3_ada.tolist())
        writer.writerow(["LASSO"] + fnr_arr_3_lasso.tolist())
        writer.writerow(["OLS"] + fnr_arr_3_mle.tolist())
        f.close()

    with open(fnr_txt_mix_pth, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if K_or_T == 1:
            writer.writerow(["Coupling Strength"] + hori_arr.tolist())
        else:
            writer.writerow(["# of cycles"] + hori_arr.tolist())
        writer.writerow(["Adaptive LASSO"] + fnr_arr_mix_ada.tolist())
        writer.writerow(["LASSO"] + fnr_arr_mix_lasso.tolist())
        writer.writerow(["OLS"] + fnr_arr_mix_mle.tolist())
        f.close()

    with open(mcc_txt_2_pth, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if K_or_T == 1:
            writer.writerow(["Coupling Strength"] + hori_arr.tolist())
        else:
            writer.writerow(["# of cycles"] + hori_arr.tolist())
        writer.writerow(["Adaptive LASSO"] + mcc_arr_2_ada.tolist())
        writer.writerow(["LASSO"] + mcc_arr_2_lasso.tolist())
        writer.writerow(["OLS"] + mcc_arr_2_mle.tolist())
        f.close()

    with open(mcc_txt_3_pth, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if K_or_T == 1:
            writer.writerow(["Coupling Strength"] + hori_arr.tolist())
        else:
            writer.writerow(["# of cycles"] + hori_arr.tolist())
        writer.writerow(["Adaptive LASSO"] + mcc_arr_3_ada.tolist())
        writer.writerow(["LASSO"] + mcc_arr_3_lasso.tolist())
        writer.writerow(["OLS"] + mcc_arr_3_mle.tolist())
        f.close()

    with open(mcc_txt_mix_pth, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if K_or_T == 1:
            writer.writerow(["Coupling Strength"] + hori_arr.tolist())
        else:
            writer.writerow(["# of cycles"] + hori_arr.tolist())
        writer.writerow(["Adaptive LASSO"] + mcc_arr_mix_ada.tolist())
        writer.writerow(["LASSO"] + mcc_arr_mix_lasso.tolist())
        writer.writerow(["OLS"] + mcc_arr_mix_mle.tolist())
        f.close()


def drawing_mcc_only(outputs, p_lst_, path_lst, path_base, sharex=False, K_or_T=1, mle=True, mle_ori=True,
                     lasso=True, ada=True):
    assert len(outputs) == len(p_lst_), "check length of outputs and p list. "
    if sharex:
        hori_arr = outputs[0]["x"]

    fig, ax = plt.subplots(nrows=len(p_lst_), ncols=3, sharex='col', sharey=True, figsize=(18, 18), dpi=100)
    for i in range(len(outputs)):
        if not sharex:
            hori_arr = outputs[i]["x"]

        if ada:
            mcc_arr_2_ada = outputs[i]["ada"][9]
            mcc_arr_3_ada = outputs[i]["ada"][10]
            mcc_arr_mix_ada = outputs[i]["ada"][11]

            ax[0, i].plot(hori_arr, mcc_arr_2_ada, label="Pairwise MCC, A. LASSO", linewidth=5, color="red")
            ax[1, i].plot(hori_arr, mcc_arr_3_ada, label="3-int MCC, A. LASSO", linewidth=5, color="red")
            ax[2, i].plot(hori_arr, mcc_arr_mix_ada, label="Mix MCC, A. LASSO", linewidth=5, color="red")

        if lasso:
            mcc_arr_2_lasso = outputs[i]["lasso"][9]
            mcc_arr_3_lasso = outputs[i]["lasso"][10]
            mcc_arr_mix_lasso = outputs[i]["lasso"][11]

            ax[0, i].plot(hori_arr, mcc_arr_2_lasso, label="Pairwise MCC, LASSO", linewidth=5, alpha=0.6, color="cyan")
            ax[1, i].plot(hori_arr, mcc_arr_3_lasso, label="3-int MCC, LASSO", linewidth=5, alpha=0.6, color="cyan")
            ax[2, i].plot(hori_arr, mcc_arr_mix_lasso, label="Mix MCC, LASSO", linewidth=5, alpha=0.6, color="cyan")

        if mle:
            mcc_arr_2_mle = outputs[i]["mle"][9]
            mcc_arr_3_mle = outputs[i]["mle"][10]
            mcc_arr_mix_mle = outputs[i]["mle"][11]

            ax[0, i].plot(hori_arr, mcc_arr_2_mle, label="Pairwise MCC, OLS with FDR control", linewidth=5,
                          color="purple")
            ax[1, i].plot(hori_arr, mcc_arr_3_mle, label="3-int MCC, OLS with FDR control", linewidth=5, color="purple")
            ax[2, i].plot(hori_arr, mcc_arr_mix_mle, label="Mix MCC, OLS with FDR control", linewidth=5, color="purple")

        if mle_ori:
            mcc_arr_2_mle_ori = outputs[i]["mle_ori"][9]
            mcc_arr_3_mle_ori = outputs[i]["mle_ori"][10]
            mcc_arr_mix_mle_ori = outputs[i]["mle_ori"][11]

            ax[0, i].plot(hori_arr, mcc_arr_2_mle_ori, label="Pairwise MCC, OLS with threshold", linewidth=5,
                          color="grey")
            ax[1, i].plot(hori_arr, mcc_arr_3_mle_ori, label="3-int MCC, OLS with threshold", linewidth=5, color="grey")
            ax[2, i].plot(hori_arr, mcc_arr_mix_mle_ori, label="Mix MCC, OLS with threshold", linewidth=5, color="grey")

        ax[0, i].spines[['right', 'top']].set_visible(False)
        ax[1, i].spines[['right', 'top']].set_visible(False)
        ax[2, i].spines[['right', 'top']].set_visible(False)

        ax[0, i].tick_params('both', length=15, width=1, which='major')
        ax[0, i].tick_params('both', length=10, width=1, which='minor')
        ax[1, i].tick_params('both', length=15, width=1, which='major')
        ax[1, i].tick_params('both', length=10, width=1, which='minor')
        ax[2, i].tick_params('both', length=15, width=1, which='major')
        ax[2, i].tick_params('both', length=10, width=1, which='minor')

        # ax[0, i].hlines(0.7, xmin=hori_arr[0], xmax=hori_arr[-1], linestyles="--")
        # ax[1, i].hlines(0.7, xmin=hori_arr[0], xmax=hori_arr[-1], linestyles="--")
        # ax[2, i].hlines(0.7, xmin=hori_arr[0], xmax=hori_arr[-1], linestyles="--")

        title_1 = f"Pairwise $p={p_lst_[i]}$"
        title_2 = f"3-int $p={p_lst_[i]}$"
        title_3 = f"Mix $p={p_lst_[i]}$"
        ax[0, i].set_title(title_1, fontsize=30)
        ax[1, i].set_title(title_2, fontsize=30)
        ax[2, i].set_title(title_3, fontsize=30)

        ax[0, i].set_ylim(-0.01, 1.01)

        paths = path_lst[i]

        mcc_txt_2_pth = paths + f"MCC_pairwise_p={p_lst_[i]}.csv"
        mcc_txt_3_pth = paths + f"MCC_3-interaction_p={p_lst_[i]}.csv"
        mcc_txt_mix_pth = paths + f"MCC_mixture_p={p_lst_[i]}.csv"

        with open(mcc_txt_2_pth, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            if K_or_T == 1:
                writer.writerow(["Coupling Strength"] + hori_arr.tolist())
            else:
                writer.writerow(["# of cycles"] + hori_arr.tolist())
            if ada:
                writer.writerow(["Adaptive LASSO"] + mcc_arr_2_ada.tolist())
            if lasso:
                writer.writerow(["LASSO"] + mcc_arr_2_lasso.tolist())
            if mle:
                writer.writerow(["OLS with FDR control"] + mcc_arr_2_mle.tolist())
            if mle_ori:
                writer.writerow(["OLS with threshold"] + mcc_arr_2_mle_ori.tolist())
            f.close()

        with open(mcc_txt_3_pth, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            if K_or_T == 1:
                writer.writerow(["Coupling Strength"] + hori_arr.tolist())
            else:
                writer.writerow(["# of cycles"] + hori_arr.tolist())
            if ada:
                writer.writerow(["Adaptive LASSO"] + mcc_arr_3_ada.tolist())
            if lasso:
                writer.writerow(["LASSO"] + mcc_arr_3_lasso.tolist())
            if mle:
                writer.writerow(["OLS with FDR control"] + mcc_arr_3_mle.tolist())
            if mle_ori:
                writer.writerow(["OLS with threshold"] + mcc_arr_3_mle_ori.tolist())
            f.close()

        with open(mcc_txt_mix_pth, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            if K_or_T == 1:
                writer.writerow(["Coupling Strength"] + hori_arr.tolist())
            else:
                writer.writerow(["# of cycles"] + hori_arr.tolist())
            if ada:
                writer.writerow(["Adaptive LASSO"] + mcc_arr_mix_ada.tolist())
            if lasso:
                writer.writerow(["LASSO"] + mcc_arr_mix_lasso.tolist())
            if mle:
                writer.writerow(["OLS with FDR control"] + mcc_arr_mix_mle.tolist())
            if mle_ori:
                writer.writerow(["OLS with threshold"] + mcc_arr_mix_mle_ori.tolist())
            f.close()

    if K_or_T == 1:
        hori_arr = np.array([0, .04, .08, .12, .16])

    # plt.rcParams["font.size"] = 36  # フォントを大きく
    plt.rcParams['font.family'] = 'Arial'
    params = {'legend.fontsize': 20,
              'legend.handlelength': 1}
    plt.rcParams.update(params)
    label_ = []
    if ada:
        label_.append("Adaptive LASSO")
    if lasso:
        label_.append("LASSO")
    if mle:
        label_.append("OLS with FDR control")
    if mle_ori:
        label_.append("OLS with threshold")
    fig.legend(ax, labels=label_, loc="upper right")
    if K_or_T == 1:
        xlabel = 'Coupling strength'
        fig.suptitle('MCC vs. Coupling strength', size=40)
    else:
        xlabel = 'Cycle number'
        fig.suptitle('MCC vs. # of cycles', size=40)
        ax[-1, 0].set_xscale("log")
        ax[-1, 1].set_xscale("log")
        ax[-1, 2].set_xscale("log")

    ax[0, 0].set_ylabel('MCC', rotation=90, fontsize=24)
    ax[0, 0].tick_params(axis='y', labelsize=18)
    ax[1, 0].set_ylabel('MCC', rotation=90, fontsize=24)
    ax[1, 0].tick_params(axis='y', labelsize=18)
    ax[2, 0].set_ylabel('MCC', rotation=90, fontsize=24)
    ax[2, 0].tick_params(axis='y', labelsize=18)

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

    path = path_base + "MCC.jpg"
    fig.savefig(path)
    if K_or_T == 1:
        path_eps = path_base + "MCC_K.eps"
    elif K_or_T == 2:
        path_eps = path_base + "MCC_T.eps"
    fig.savefig(path_eps, format="eps", bbox_inches=None)


def create_task_Fig3(conn_seed_, coup_seed_, noise_arr, natfreq_arr, K, T, pre_conn_2_arr=None, pre_conn_3_arr=None,
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
                                       starts_from=st_fr, inf_last=inf_l, old_legacy=True)

        while not check_larger_than_limit(two_model, ops_limit):
            new_seed = np.random.randint(0, 1e10)
            two_model = GeneralInteraction(coupling2=K, coupling3=0, dt=dt, T=T, natfreqs=natfreq_arr,
                                           with_noise=True,
                                           noise_sth=noise_sth, normalize=True, conn2=p_, all_connected=all_connected,
                                           conn_seed=new_seed, coup_seed=coup_arr[i], noise_seed=noise_arr[0],
                                           starts_from=st_fr, inf_last=inf_l, old_legacy=True)

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
                                         noise_seed=noise_arr[1], starts_from=st_fr, inf_last=inf_l, old_legacy=True)

        while not check_larger_than_limit(three_model, ops_limit):
            new_seed = np.random.randint(0, 1e10)
            three_model = GeneralInteraction(coupling2=0, coupling3=K * ratio, dt=dt, T=T, natfreqs=natfreq_arr,
                                             with_noise=True, noise_sth=noise_sth, normalize=True, conn3=p_,
                                             all_connected=False, conn_seed=new_seed, coup_seed=coup_arr[i],
                                             noise_seed=noise_arr[1], starts_from=st_fr, inf_last=inf_l,
                                             old_legacy=True)

        if conn_arr3[i] != three_model.conn_seed:
            conn_arr3[i] = three_model.conn_seed
            real_conn_3_seed[i] = three_model.conn_seed
        three_lst.append(three_model)
    task_lst.append(three_lst)
    conn_3_seed = merge_seed_array(conn_arr3, real_conn_3_seed)
    for i in range(trial_num):
        conn_mat2 = task_lst[0][i].conn_mat2
        conn_mat3 = task_lst[1][i].conn_mat3
        reduced_2 = reduce_conn_2(conn_mat2, 0.5, reduce_seed=reduce_seed)
        reduced_3 = reduce_conn_3(conn_mat3, 0.5, reduce_seed=reduce_seed)
        reduced_2_coup = reduced_2 * (K / (p_ * len(natfreq_arr)))
        reduced_3_coup = reduced_3 * (K * ratio / (p_ * len(natfreq_arr) * len(natfreq_arr)))

        mix_model = GeneralInteraction(dt=dt, T=T, natfreqs=natfreq_arr,
                                       with_noise=True, noise_sth=noise_sth, conn=p_ / 2,
                                       pre_conn2=reduced_2, pre_conn3=reduced_3, pre_coup2=reduced_2_coup,
                                       pre_coup3=reduced_3_coup,
                                       noise_seed=noise_arr[2], starts_from=st_fr, inf_last=inf_l)

        mix_lst.append(mix_model)
    task_lst.append(mix_lst)
    return task_lst, conn_2_seed, conn_3_seed


def check_larger_than_limit(model: GeneralInteraction, ops_limit=1.0):
    act_mat = model.run()
    op = [model.phase_coherence(vec) for vec in act_mat.T]
    ops_avg = np.average(op)
    if ops_avg <= ops_limit:
        return True
    else:
        return False


def merge_seed_array(original_arr, output_arr):
    assert len(original_arr) == len(output_arr), "Lengths of two arrays needs to be equal. "
    copy_arr = np.copy(original_arr)
    for i in range(len(output_arr)):
        if output_arr[i] != original_arr[i]:
            copy_arr[i] = output_arr[i]

    return copy_arr


def extract_nonzero_elements(arr):
    return arr[arr != 0].ravel()


if __name__ == "__main__":
    natfreq_seed = 98765
    conn_seed = 19961102
    # conn_seed = None
    coup_seed = None
    noise_seed = 20116991
    # noise_seed = None

    start_datetime = datetime.now()
    p_lst = [0.05, 0.10, 0.15]
    # two_arr = np.load("Fig3data/img/20240913/")
    # three_arr = np.load("Fig3data/img/20240913/")
    main_for_Fig3(natfreq_seed, conn_seed, coup_seed, noise_seed, K_or_T=2, pre_conn_2_lst=None, pre_conn_3_lst=None,
                  p_lst_=p_lst, draw=2, trial_num=10, mle_ori=False, ols=True, lasso=True, ada=True, mle_threshold=0.25)
    plt.show()

    now = datetime.now()
    duration = now - start_datetime
    print("Duration is =", duration)
