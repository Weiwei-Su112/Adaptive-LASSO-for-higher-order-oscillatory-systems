from GeneralInteraction import *
import pickle

n_nodes = 12

conn2 = 0.05
conn3 = 0.01
conn4 = 0.005
nor_2 = conn2 * n_nodes
nor_3 = conn3 * n_nodes * n_nodes
nor_4 = conn4 * n_nodes * n_nodes * n_nodes

natfreqs = np.random.default_rng(5463).normal(loc=1, scale=0.1, size=n_nodes)


def gen_4_coup(n_nodes: int, prob: float, seed: int):
    rng = np.random.default_rng(seed)
    four_conn_ = np.zeros(shape=(n_nodes, n_nodes, n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            for k in range(j+1, n_nodes):
                for l in range(k+1, n_nodes):
                    indicator = rng.random()
                    if indicator < prob:
                        four_conn_[i][j][k][l] = 1

    return four_conn_


# two_conn[2, 0] = 1  # K_53
# two_coup[2, 0] = 0.1
# two_conn[1, 4] = 1  # K_42
# two_coup[1, 4] = 0.1
#
# three_conn[3, 1, 2] = 1  # K_423
# three_coup[3, 1, 2] = 0.3
# three_conn[1, 3, 4] = 1  # K_245
# three_coup[1, 3, 4] = 0.3
# three_conn[0, 2, 3] = 1  # K_134
# three_coup[0, 2, 3] = 0.3
#
# four_conn[0, 1, 2, 3] = 1  # K_1234
# four_coup[0, 1, 2, 3] = 0.5
# four_conn[3, 0, 2, 4] = 1  # K_4135
# four_coup[3, 0, 2, 4] = 0.5

rng_seed = np.random.default_rng(3928310123)
conn4_seed_rng = np.random.default_rng(32942030131)
indi = False
while not indi:
    conn_seed = rng_seed.integers(0, 1e12)
    conn_seed_sub = conn4_seed_rng.integers(0, 1e12)
    noise_seed = 9543234

    # conn_seed = 393058876878
    # conn_seed_sub = 880825750742
    # noise_seed = 5644312
    # indi = True

    # conn_seed = 602233353736
    # conn_seed_sub = 432087895040
    # noise_seed = 563543
    # indi = True

    conn_seed = 107889519366
    conn_seed_sub = 926534431405
    noise_seed = 9543234
    indi = True

    four_conn = gen_4_coup(n_nodes, conn4, conn_seed_sub)
    four_coup = 0.1 * four_conn
    model_ = GeneralInteraction(dt=0.02, T=900, natfreqs=natfreqs, with_noise=True, coupling2=0.1,
                                coupling3=0.1,
                                random_coup2=False, random_coup3=False, conn_seed=conn_seed,
                                noise_sth=0.2, normalize=True, all_connected=False,
                                noise_seed=noise_seed, starts_from=0, inf_last=1, type2=5, type3=5,
                                pre_conn4=four_conn, pre_coup4=four_coup, conn4=conn4, conn3=conn3, conn2=conn2)

    if np.sum(model_.conn_mat3) <= 6 or np.sum(four_conn) < 10 or np.sum(model_.conn_mat2) <= 6:
        continue

    ols_results, lasso_results, ada_results = model_.demo_solve(mle=True, lasso=True, ada=True, print_result=False, op_illu=True)
    all_res = model_.conn_criteria_base(ada_results=ada_results, lasso_results=lasso_results,
                                        mle_or_ols_results=ols_results)
    ada_conn = all_res["ada"]
    lasso_conn = all_res["lasso"]
    ols_conn = all_res["mle"]
    if ada_conn["TP_4"] < 4:
        continue
    mcc_ada = model_.MCC_easy(TP=ada_conn["TP"], TN=ada_conn["TN"], FP=ada_conn["FP"], FN=ada_conn["FN"])
    mcc_lasso = model_.MCC_easy(TP=lasso_conn["TP"], TN=lasso_conn["TN"], FP=lasso_conn["FP"], FN=lasso_conn["FN"])
    mcc_ols = model_.MCC_easy(TP=ols_conn["TP"], TN=ols_conn["TN"], FP=ols_conn["FP"], FN=ols_conn["FN"])
    # print("MCC is: ", mcc_ada)
    if mcc_ada >= 0.75:
        print("MCC is: ", mcc_ada)
        print("Seed is:", conn_seed, "and", conn_seed_sub)
        indi = True

now = datetime.now()
date = now.strftime("%Y%m%d")

tm_string = now.strftime("%H%M%S") + "_" + str(0)
file_path = "../../4data/" + date + "/"
os.makedirs(file_path, exist_ok=True)
base_path = "../../4data/" + date + "/" + tm_string + "/"
os.mkdir(base_path)

root_path = base_path

file_name0 = root_path + "natfreqs.csv"
file_name1 = root_path + "pairwise.csv"
file_name2 = root_path + "3-interaction.csv"
file_name3 = root_path + "4-interaction.csv"
file_name_confusion = root_path + "confusion_values.csv"
file_seed = root_path + "seed.txt"
demo_coup2 = model_.coupling2
demo_coup3 = model_.coupling3
demo_coup4 = model_.coupling4

header0 = ["i", "Real", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - OLS with FDR"]
header2 = ["i", "j", "Real", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - OLS with FDR"]
header3 = ["i", "j", "k", "Real", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - OLS with FDR"]
header4 = ["i", "j", "k", "l", "Real", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - OLS with FDR"]
header_confusion = ["Confusion Values", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - OLS with FDR"]

pkl_ada = base_path + "ada.pkl"

natfreq_lst = []
lst_2 = []
lst_3 = []
lst_4 = []

with open(pkl_ada, "wb") as f:
    pickle.dump(ada_results, f)

for i in range(model_.n_nodes):
    others_lst = np.delete(model_.all_nodes, i)
    more_others_lst = model_.make_more_others_lst(others_lst, i)

    demo_lst2 = ["Real-2", model_.natfreqs[i]]
    for j in others_lst:
        demo_lst2.append(demo_coup2[j][i])
    real_2 = demo_lst2[2:]

    demo_lst3 = ["Real-3"]
    demo_lst4 = ["Real-4"]
    for j in others_lst:
        if j == others_lst[-1]:
            break
        elif j < i:
            for k in more_others_lst[j]:
                demo_lst3.append(demo_coup3[i][j][k])
                arr = more_others_lst[j]
                others4 = arr[arr > k]
                for l in others4:
                    demo_lst4.append(demo_coup4[i][j][k][l])
        else:
            for k in more_others_lst[j - 1]:
                demo_lst3.append(demo_coup3[i][j][k])
                arr = more_others_lst[j - 1]
                others4 = arr[arr > k]
                for l in others4:
                    demo_lst4.append(demo_coup4[i][j][k][l])
    real_3 = demo_lst3[1:]
    real_4 = demo_lst4[1:]

    ada_2 = ada_results[i]["2"]
    ada_3 = ada_results[i]["3"]
    ada_4 = ada_results[i]["4"]
    lasso_2 = lasso_results[i]["2"]
    lasso_3 = lasso_results[i]["3"]
    lasso_4 = lasso_results[i]["4"]
    ols_2 = ols_results[i]["2"]
    ols_3 = ols_results[i]["3"]
    ols_4 = ols_results[i]["4"]

    counter_2 = 0
    col_names2 = ["", "natural frequencies_" + str(i + 1)]
    for index in others_lst:
        name = "k_" + str(index + 1) + str(i + 1)
        col_names2.append(name)
        inputs = [str(i + 1), str(index + 1), real_2[counter_2] * nor_2, ada_2[counter_2] * nor_2,
                  lasso_2[counter_2] * nor_2, ols_2[counter_2] * nor_2]
        # for trial sent to Hata-san
        # inputs = [str(i + 1), str(index + 1), real_2[counter_2]]
        lst_2.append(inputs)
        counter_2 += 1

    counter_3 = 0
    counter_4 = 0
    col_names3 = [""]
    col_names4 = [""]
    for index in others_lst:
        if index == others_lst[-1]:
            break
        elif index < i:
            for inde2 in more_others_lst[index]:
                name = "k_" + str(index + 1) + str(inde2 + 1) + str(i + 1)
                col_names3.append(name)
                inputs = [str(i + 1), str(index + 1), str(inde2 + 1), real_3[counter_3] * nor_3,
                          ada_3[counter_3] * nor_3, lasso_3[counter_3] * nor_3, lasso_3[counter_3] * nor_3]
                arr = more_others_lst[index]
                others4 = arr[arr > inde2]
                for l in others4:
                    name = "k_" + str(index + 1) + str(inde2 + 1) + str(l + 1) + str(i + 1)
                    col_names4.append(name)
                    inputs_sub = [str(i + 1), str(index + 1), str(inde2 + 1), str(l + 1), real_4[counter_4] * nor_4,
                              ada_4[counter_4] * nor_4, lasso_4[counter_4] * nor_4, ols_4[counter_4] * nor_4]
                    lst_4.append(inputs_sub)
                    counter_4 += 1
                # for trial sent to Hata-san
                # inputs = [str(i + 1), str(index + 1), str(inde2 + 1), real_3[counter_3]]
                lst_3.append(inputs)
                counter_3 += 1
        else:
            for inde2 in more_others_lst[index - 1]:
                name = "k_" + str(index + 1) + str(inde2 + 1) + str(i + 1)
                col_names3.append(name)
                inputs = [str(i + 1), str(index + 1), str(inde2 + 1), real_3[counter_3] * nor_3,
                          ada_3[counter_3] * nor_3, lasso_3[counter_3] * nor_3, ols_3[counter_3] * nor_3]
                arr = more_others_lst[index - 1]
                others4 = arr[arr > inde2]
                for l in others4:
                    name = "k_" + str(index + 1) + str(inde2 + 1) + str(l + 1) + str(i + 1)
                    col_names4.append(name)
                    inputs_sub = [str(i + 1), str(index + 1), str(inde2 + 1), str(l + 1), real_4[counter_4] * nor_4,
                                  ada_4[counter_4] * nor_4, lasso_4[counter_4] * nor_4, ols_4[counter_4] * nor_4]
                    lst_4.append(inputs_sub)
                    counter_4 += 1
                # for trial sent to Hata-san
                # inputs = [str(i + 1), str(index + 1), str(inde2 + 1), real_3[counter_3]]
                lst_3.append(inputs)
                counter_3 += 1

    natfreq_lst.append([str(i + 1), model_.natfreqs[i], ada_results[i]["natfreq"],
                        lasso_results[i]["natfreq"], ols_results[i]["natfreq"]])
    # natfreq_lst.append([str(i + 1), model_.natfreqs[i] % (2 * np.pi)])

with open(file_name_confusion, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_confusion)

    mse_ada = model_.mse_and_r2_combined(ada_results)[0]
    fpr_ada = ada_conn["FP"] / (ada_conn["TN"] + ada_conn["FP"])
    fnr_ada = ada_conn["FN"] / (ada_conn["TP"] + ada_conn["FN"])
    mse_lasso = model_.mse_and_r2_combined(lasso_results)[0]
    fpr_lasso = lasso_conn["FP"] / (lasso_conn["TN"] + lasso_conn["FP"])
    fnr_lasso = lasso_conn["FN"] / (lasso_conn["TP"] + lasso_conn["FN"])
    mse_ols = model_.mse_and_r2_combined(ols_results)[0]
    fpr_ols = ols_conn["FP"] / (ols_conn["TN"] + ols_conn["FP"])
    fnr_ols = ols_conn["FN"] / (ols_conn["TP"] + ols_conn["FN"])

    writer.writerow(["TP_2", ada_conn["TP_2"], lasso_conn["TP_2"], ols_conn["TP_2"]])
    writer.writerow(["TN_2", ada_conn["TN_2"], lasso_conn["TN_2"], ols_conn["TN_2"]])
    writer.writerow(["FP_2", ada_conn["FP_2"], lasso_conn["FP_2"], ols_conn["FP_2"]])
    writer.writerow(["FN_2", ada_conn["FN_2"], lasso_conn["FN_2"], ols_conn["FN_2"]])
    writer.writerow(["TP_3", ada_conn["TP_3"], lasso_conn["TP_3"], ols_conn["TP_3"]])
    writer.writerow(["TN_3", ada_conn["TN_3"], lasso_conn["TN_3"], ols_conn["TN_3"]])
    writer.writerow(["FP_3", ada_conn["FP_3"], lasso_conn["FP_3"], ols_conn["FP_3"]])
    writer.writerow(["FN_3", ada_conn["FN_3"], lasso_conn["FN_3"], ols_conn["FN_3"]])
    writer.writerow(["TP_4", ada_conn["TP_4"], lasso_conn["TP_4"], ols_conn["TP_4"]])
    writer.writerow(["TN_4", ada_conn["TN_4"], lasso_conn["TN_4"], ols_conn["TN_4"]])
    writer.writerow(["FP_4", ada_conn["FP_4"], lasso_conn["FP_4"], ols_conn["FP_4"]])
    writer.writerow(["FN_4", ada_conn["FN_4"], lasso_conn["FN_4"], ols_conn["FN_4"]])
    writer.writerow(["TP", ada_conn["TP"], lasso_conn["TP"], ols_conn["TP"]])
    writer.writerow(["TN", ada_conn["TN"], lasso_conn["TN"], ols_conn["TN"]])
    writer.writerow(["FP", ada_conn["FP"], lasso_conn["FP"], ols_conn["FP"]])
    writer.writerow(["FN", ada_conn["FN"], lasso_conn["FN"], ols_conn["FN"]])
    writer.writerow(["MCC", mcc_ada, mcc_lasso, mcc_ols])
    writer.writerow(["MSE", mse_ada, mse_lasso, mse_ols])
    writer.writerow(["FPR", fpr_ada, fpr_lasso, fpr_ols])
    writer.writerow(["FNR", fnr_ada, fnr_lasso, fnr_ols])

    f.close()

with open(file_name0, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header0)
    writer.writerows(natfreq_lst)
    # somehow for-loop every row with the information
    f.close()

with open(file_name1, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header2)
    writer.writerows(lst_2)
    # somehow for-loop every row with the information
    f.close()

with open(file_name2, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header3)
    writer.writerows(lst_3)
    # somehow for-loop every row with the information
    f.close()

with open(file_name3, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header4)
    writer.writerows(lst_4)
    # somehow for-loop every row with the information
    f.close()

with open(file_seed, 'w', encoding='UTF8', newline='') as f:
    f.write(f"seed is conn: {conn_seed} and 4-conn{conn_seed_sub}, noise seed={noise_seed}")
    f.close()

os.chdir(root_path)

