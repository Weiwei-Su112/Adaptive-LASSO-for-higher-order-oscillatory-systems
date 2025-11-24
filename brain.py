from GeneralInteraction import *
import pandas as pd
import pickle


def mse_simple(demo: np.ndarray, estimated: np.ndarray, tp_only=False):
    if not tp_only:
        rss = ((demo - estimated) ** 2).sum()
    else:
        test = np.logical_and(demo != 0, estimated != 0)
        avg_calc = demo[test]
        rss = ((avg_calc - estimated[test]) ** 2).sum()

    return 1 / len(demo) * rss


def threshold_lognormal(mean, sigma, rng_: np.random._generator.Generator, threshold=None):
    num = rng_.lognormal(mean=mean, sigma=sigma)

    if type(threshold) == float:
        while num < threshold:
            num = rng_.lognormal(mean=mean, sigma=sigma)

    return num


def count_mutual_pairs(adj):
    n = adj.shape[0]
    mutual = 0
    for i in range(n):
        for j in range(i + 1, n):  # j > i to avoid double counting
            if adj[i][j] and adj[j][i]:
                mutual += 1
    return mutual


def count_mutual_pairs_vectorized(adj):
    mutual_matrix = np.logical_and(adj, adj.T)
    upper_triangle = np.triu(mutual_matrix, k=1)
    return np.sum(upper_triangle)


def count_unique_directed_triangles(adj):
    n = adj.shape[0]
    counted_triangles = set()
    count = 0

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                # 判断是否有三角形 i->j, j->k, k->i
                if adj[i][j] and adj[j][k] and adj[k][i]:
                    # 规范化三角形的节点顺序：按节点编号排序
                    triangle_nodes = tuple(sorted([i, j, k]))
                    if triangle_nodes not in counted_triangles:
                        counted_triangles.add(triangle_nodes)
                        count += 1
    return count


def lognormal_pairwise(pairwise_adj, seed):
    n = pairwise_adj.shape[0]
    coup = np.zeros((n, n))

    rng = np.random.default_rng(seed)

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            elif pairwise_adj[i][j] == 1:
                coup[i][j] = threshold_lognormal(mean=-2, sigma=0.5, rng_=rng, threshold=0.05)

    return coup


def directed_triangle_tensor_symmetric(adj, seed: None or int = None, poss=1.0):
    n = adj.shape[0]
    tensor = np.zeros((n, n, n), dtype=int)
    coup = np.zeros((n, n, n))
    counted_triangles = set()

    if seed is None:
        seed = np.random.randint(0, 1e8)
        print("The coupling seed is ", seed)
        assert type(seed) == int, "Boom"
    rng = np.random.default_rng(seed)
    indicator_rng = np.random.default_rng(rng.integers(0, 1e12))

    conn_gen_rng = np.random.default_rng(102934432)

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                # Check for i -> j -> k -> i
                triangle1 = adj[i][j] and adj[j][k] and adj[k][i]
                # Check for i -> k -> j -> i
                triangle2 = adj[i][k] and adj[k][j] and adj[j][i]
                triangle_nodes = tuple(sorted([i, j, k]))

                if (triangle1 or triangle2) and triangle_nodes not in counted_triangles:
                    counted_triangles.add(triangle_nodes)
                    indicator = indicator_rng.random()
                    if indicator < poss:
                        conn_gen_num = conn_gen_rng.random()
                        # if adj[i][j] != 0 and adj[j][k] != 0 and adj[k][i] != 0:
                        #     strength = (adj[i][j] + adj[j][k] + adj[k][i]) / 3
                        # elif adj[i][k] != 0 and adj[k][j] != 0 and adj[j][i] != 0:
                        #     strength = (adj[i][k] + adj[k][j] + adj[j][i]) / 3
                        # else:
                        #     strength = 0
                        if conn_gen_num < 1 / 3 and j < k:
                            tensor[i][j][k] = 1  # only one entry per (i, j, k) with j < k
                            coup[i][j][k] = threshold_lognormal(mean=-2, sigma=0.5, rng_=rng, threshold=0.05)
                            # coup[i][j][k] = strength
                        elif conn_gen_num < 1 / 3 and k < j:
                            tensor[i][k][j] = 1  # only one entry per (i, j, k) with j < k
                            coup[i][k][j] = threshold_lognormal(mean=-2, sigma=0.5, rng_=rng, threshold=0.05)
                            # coup[i][k][j] = strength
                        elif conn_gen_num < 2 / 3 and k < i:
                            tensor[j][k][i] = 1  # only one entry per (i, j, k) with j < k
                            coup[j][k][i] = threshold_lognormal(mean=-2, sigma=0.5, rng_=rng, threshold=0.05)
                            # coup[j][k][i] = strength
                        elif conn_gen_num < 2 / 3 and i < k:
                            tensor[j][i][k] = 1  # only one entry per (i, j, k) with j < k
                            coup[j][i][k] = threshold_lognormal(mean=-2, sigma=0.5, rng_=rng, threshold=0.05)
                            # coup[j][i][k] = strength
                        elif conn_gen_num > 2 / 3 and i < j:
                            tensor[k][i][j] = 1  # only one entry per (i, j, k) with j < k
                            coup[k][i][j] = threshold_lognormal(mean=-2, sigma=0.5, rng_=rng)
                            # coup[k][i][j] = strength
                        elif conn_gen_num > 2 / 3 and j < i:
                            tensor[k][j][i] = 1  # only one entry per (i, j, k) with j < k
                            coup[k][j][i] = threshold_lognormal(mean=-2, sigma=0.5, rng_=rng)
                            # coup[k][j][i] = strength
                        # tensor[i][j][k] = 1  # only one entry per (i, j, k) with j < k
                        # coup[i][j][k] = rng.lognormal(mean=-2, sigma=0.5)
                        #### thresholding for the lognormal? 1.5 or 2?

    return tensor, coup


mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']

start_datetime = datetime.now()
T = int(400 * np.pi)
dt = 0.02
starts_from = 0
cut_time = int(T * starts_from)
inf_last = 1

plt_x_axis = np.arange(-cut_time, T - cut_time, dt)

now = datetime.now()
date = now.strftime("%Y%m%d")

tm_string = now.strftime("%H%M%S") + "_" + str(0)
file_path = "../../Fig5data/" + date + "/"
base_path = "../../Fig5data/" + date + "/" + tm_string + "/"

# 001, 062, 038, 049, 026
df = pd.read_csv("structural_connectivity_matrices/S026.csv", header=None)
coup2 = df.to_numpy()
conn2 = np.copy(coup2)
conn2[conn2 < 0.05] = 0
conn2[conn2 != 0] = 1
coup2 = lognormal_pairwise(conn2, seed=948521)
print("Mutually connected pairs: ", count_mutual_pairs(conn2))
print(np.sum(conn2))
# coup2[coup2 < 0.05] = 0
# coup2 = coup2 * 1
# nonzero_avg = coup2[coup2 != 0].mean()
# print("The nonzero pairwise mean is: ", nonzero_avg)
a = np.sum(conn2) / 90 / 89
conn3, coup3 = directed_triangle_tensor_symmetric(coup2, seed=32007965, poss=1)
# conn3, coup3 = directed_triangle_tensor_symmetric(conn2, seed=3200792, poss=1)
count_tri = count_unique_directed_triangles(conn2)
b = np.sum(conn3) / (90 * 89 * 88 / 2)
c = np.sum(conn3)
# print(coup2)
print(a)
print(b)
print(c)
print("# of unique triangles are: ", count_tri)
test2 = np.copy(coup2)
test3 = np.copy(coup3)
test2 = test2[test2 != 0]
test3 = test3[test3 != 0]
print("min of pairwise before normalization: ", test2.min())
print("max of pairwise before normalization: ", test2.max())
print("min of 3-body before normalization: ", test3.min())
print("max of 3-body before normalization: ", test3.max())

plt.hist(test2, bins=10, edgecolor='black')
# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Pairwise coupling strength before normalization - brain')
plt.show()

plt.hist(test3, bins=10, edgecolor='black')
# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('3-body coupling strength before normalization - brain')
plt.show()

show_normalization_value = True
if show_normalization_value:
    nor_2 = a * 90
    nor_3 = b * 90 * 90
else:
    nor_2 = 1
    nor_3 = 1

natfreqs = np.random.default_rng(9876).normal(size=90, loc=1, scale=0.1) % (2 * np.pi)
model_ = GeneralInteraction(dt=dt, T=T, n_nodes=90, noise_sth=0.2, noise_seed=302991, natfreqs=natfreqs,
                            pre_conn2=conn2, pre_coup2=coup2, pre_conn3=conn3, normalize=True,
                            pre_coup3=coup3, conn2=a, conn3=b)
print(np.sum(model_.conn_mat3))
print(np.nonzero(model_.coupling3))
test2 = np.copy(model_.coupling2)
test3 = np.copy(model_.coupling3)
test2 = test2[test2 != 0]
test3 = test3[test3 != 0]
print("min of pairwise after normalization: ", test2.min())
print("max of pairwise after normalization: ", test2.max())
print("min of 3-body after normalization: ", test3.min())
print("max of 3-body after normalization: ", test3.max())
# act_mat = model_.run()
# print(avg_phase_coherence(act_mat, how_last=0))

# mpl.rcParams['font.size'] = 28.0
#
# fig, ax = plt.subplots(figsize=(16, 7), dpi=50)
# saver = (fig, ax)
# __, _ = plot_phase_coherence(act_mat, color="blue", outer_fig=saver)
# ax.set_title(f"Avg. order parameter {avg_phase_coherence(act_mat, how_last=0)}")
# plt.show()
# # ==============================================================================

os.makedirs(file_path, exist_ok=True)
os.mkdir(base_path)

root_path = base_path
# image_path = base_path + "order_param.png"
# fig.savefig(image_path)/

file_name0 = root_path + "natfreqs.csv"
file_name1 = root_path + "pairwise.csv"
file_name2 = root_path + "3-interaction.csv"
file_name_confusion = root_path + "confusion_values.csv"
demo_coup2 = model_.coupling2
demo_coup3 = model_.coupling3

header0 = ["i", "Real", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - OLS"]
header2 = ["i", "j", "Real", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - OLS"]
header3 = ["i", "j", "k", "Real", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - OLS"]
header_confusion = ["Confusion Values", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - OLS"]

# for trial sent to Hata-san
header0 = ["i", "Real"]
header2 = ["i", "j", "Real"]
header3 = ["i", "j", "k", "Real"]
header_confusion = ["Confusion Values"]

natfreq_lst = []
lst_2 = []
lst_3 = []

# ols_results, lasso_results, ada_results = model_.demo_solve_one_node(mle=True, ada=True, lasso=True)

pkl_ols = base_path + "ols.pkl"
pkl_lasso = base_path + "lasso.pkl"
pkl_ada = base_path + "ada.pkl"

# with open(pkl_ols, "wb") as f:
#     pickle.dump(ols_results, f)
#
# with open(pkl_lasso, "wb") as f:
#     pickle.dump(lasso_results, f)
#
# with open(pkl_ada, "wb") as f:
#     pickle.dump(ada_results, f)

with open("../../Fig5data/test/S026_2/ols.pkl", "rb") as f:
    ols_results = pickle.load(f)

with open("../../Fig5data/test/S026_2/lasso.pkl", "rb") as f:
    lasso_results = pickle.load(f)

with open("../../Fig5data/test/S026_2/ada.pkl", "rb") as f:
    ada_results = pickle.load(f)

mse_ada = model_.mse_and_r2_combined(ada_results)[0]
mse_lasso = model_.mse_and_r2_combined(lasso_results)[0]
mse_ols = model_.mse_and_r2_combined(ols_results)[0]

for i in range(model_.n_nodes):
    others_lst = np.delete(model_.all_nodes, i)
    more_others_lst = model_.make_more_others_lst(others_lst, i)

    demo_lst2 = ["Real-2", model_.natfreqs[i]]
    for j in others_lst:
        demo_lst2.append(demo_coup2[j][i])
    real_2 = demo_lst2[2:]

    demo_lst3 = ["Real-3"]
    for j in others_lst:
        if j == others_lst[-1]:
            break
        elif j < i:
            for k in more_others_lst[j]:
                demo_lst3.append(demo_coup3[i][j][k])
        else:
            for k in more_others_lst[j - 1]:
                demo_lst3.append(demo_coup3[i][j][k])
    real_3 = demo_lst3[1:]

    ada_2 = ada_results[i]["2"]
    ada_3 = ada_results[i]["3"]
    lasso_2 = lasso_results[i]["2"]
    lasso_3 = lasso_results[i]["3"]
    ols_2 = ols_results[i]["2"]
    ols_3 = ols_results[i]["3"]

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
    col_names3 = [""]
    for index in others_lst:
        if index == others_lst[-1]:
            break
        elif index < i:
            for inde2 in more_others_lst[index]:
                name = "k_" + str(index + 1) + str(inde2 + 1) + str(i + 1)
                col_names3.append(name)
                inputs = [str(i + 1), str(index + 1), str(inde2 + 1), real_3[counter_3] * nor_3,
                          ada_3[counter_3] * nor_3, lasso_3[counter_3] * nor_3, ols_3[counter_3] * nor_3]
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
                # for trial sent to Hata-san
                # inputs = [str(i + 1), str(index + 1), str(inde2 + 1), real_3[counter_3]]
                lst_3.append(inputs)
                counter_3 += 1

    natfreq_lst.append([str(i + 1), model_.natfreqs[i], ada_results[i]["natfreq"], lasso_results[i]["natfreq"],
                        ols_results[i]["natfreq"]])
    # natfreq_lst.append([str(i + 1), model_.natfreqs[i] % (2 * np.pi)])

all_res = model_.conn_criteria_base(ada_results=ada_results, lasso_results=lasso_results,
                                    mle_or_ols_results=ols_results, threshold_low=0)

with open(file_name_confusion, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_confusion)

    ada_conn = all_res["ada"]
    lasso_conn = all_res["lasso"]
    ols_conn = all_res["mle"]
    mcc_ada = model_.MCC_easy(TP=ada_conn["TP"], TN=ada_conn["TN"], FP=ada_conn["FP"], FN=ada_conn["FN"])
    mcc_lasso = model_.MCC_easy(TP=lasso_conn["TP"], TN=lasso_conn["TN"], FP=lasso_conn["FP"], FN=lasso_conn["FN"])
    mcc_ols = model_.MCC_easy(TP=ols_conn["TP"], TN=ols_conn["TN"], FP=ols_conn["FP"], FN=ols_conn["FN"])
    mse_ada = model_.mse_and_r2_combined(ada_results)[0]
    mse_lasso = model_.mse_and_r2_combined(lasso_results)[0]
    mse_ols = model_.mse_and_r2_combined(ols_results)[0]
    mcc_ada = model_.MCC_easy(TP=ada_conn["TP"], TN=ada_conn["TN"], FP=ada_conn["FP"], FN=ada_conn["FN"])
    mcc_lasso = model_.MCC_easy(TP=lasso_conn["TP"], TN=lasso_conn["TN"], FP=lasso_conn["FP"], FN=lasso_conn["FN"])
    mcc_ols = model_.MCC_easy(TP=ols_conn["TP"], TN=ols_conn["TN"], FP=ols_conn["FP"], FN=ols_conn["FN"])
    fpr_ada = ada_conn["FP"] / (ada_conn["TN"] + ada_conn["FP"])
    fpr_lasso = lasso_conn["FP"] / (lasso_conn["TN"] + lasso_conn["FP"])
    fpr_ols = ols_conn["FP"] / (ols_conn["TN"] + ols_conn["FP"])
    fnr_ada = ada_conn["FN"] / (ada_conn["TP"] + ada_conn["FN"])
    fnr_lasso = lasso_conn["FN"] / (lasso_conn["TP"] + lasso_conn["FN"])
    fnr_ols = ols_conn["FN"] / (ols_conn["TP"] + ols_conn["FN"])

    writer.writerow(["TP_2", ada_conn["TP_2"], lasso_conn["TP_2"], ols_conn["TP_2"]])
    writer.writerow(["TN_2", ada_conn["TN_2"], lasso_conn["TN_2"], ols_conn["TN_2"]])
    writer.writerow(["FP_2", ada_conn["FP_2"], lasso_conn["FP_2"], ols_conn["FP_2"]])
    writer.writerow(["FN_2", ada_conn["FN_2"], lasso_conn["FN_2"], ols_conn["FN_2"]])
    writer.writerow(["TP_3", ada_conn["TP_3"], lasso_conn["TP_3"], ols_conn["TP_3"]])
    writer.writerow(["TN_3", ada_conn["TN_3"], lasso_conn["TN_3"], ols_conn["TN_3"]])
    writer.writerow(["FP_3", ada_conn["FP_3"], lasso_conn["FP_3"], ols_conn["FP_3"]])
    writer.writerow(["FN_3", ada_conn["FN_3"], lasso_conn["FN_3"], ols_conn["FN_3"]])
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

os.chdir(root_path)

now = datetime.now()
duration = now - start_datetime
print("Duration is =", duration)

