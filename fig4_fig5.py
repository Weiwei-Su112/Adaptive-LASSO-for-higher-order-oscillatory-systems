from GeneralInteraction import *
import pandas
from Fig_4 import add_identity, shrink_two_arr


def r2_simple(demo: np.ndarray, estimated: np.ndarray, tp_only=False):
    if not tp_only:
        avg = np.average(demo)
        tss = ((demo - avg) ** 2).sum()
        rss = ((demo - estimated) ** 2).sum()
    else:
        test = np.logical_and(demo != 0, estimated != 0)
        avg_calc = demo[test]
        avg = np.average(avg_calc)
        tss = ((avg_calc - avg) ** 2).sum()
        rss = ((avg_calc - estimated[test]) ** 2).sum()

    # for mse
    # return 1 / len(demo) * rss
    return 1 - rss / tss


def mse_simple(demo: np.ndarray, estimated: np.ndarray, tp_only=False):
    if not tp_only:
        rss = ((demo - estimated) ** 2).sum()
    else:
        test = np.logical_and(demo != 0, estimated != 0)
        avg_calc = demo[test]
        rss = ((avg_calc - estimated[test]) ** 2).sum()

    return 1 / len(demo) * rss


def fpr_simple(fp, tn):
    return fp/(fp + tn)


base_path = "../../Fig5data/test/S038_2/"

path = base_path + "3-interaction.csv"
df = pandas.read_csv(path, usecols=[3, 4, 5])
real_3 = df["Real"]
ada_3 = df["Estimate - Ada. LASSO"]
# ada_3 = df["Estimate - LASSO"]
r2_3 = r2_simple(real_3, ada_3)

path = base_path + "pairwise.csv"
df = pandas.read_csv(path, usecols=[2, 3, 4])
real_2 = df["Real"]
ada_2 = df["Estimate - Ada. LASSO"]
# ada_2 = df["Estimate - LASSO"]
r2_2 = r2_simple(real_2, ada_2)

# make a histogram
# real_2_hist = real_2[real_2 != 0] * (0.005 * 100)
real_2_hist = real_2[real_2 != 0]
plt.hist(real_2_hist, bins=10, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Pairwise coupling strength before normalization - Fig. 5')
plt.show()

# real_3_hist = real_3[real_3 != 0] * (0.0001 * 100 * 100)
real_3_hist = real_3[real_3 != 0]
plt.hist(real_3_hist, bins=10, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('3-body coupling strength before normalization - Fig. 5')
plt.show()

real = np.concatenate((real_2, real_3))
ada = np.concatenate((ada_2, ada_3))
TP_values = np.sum(np.logical_and(real != 0, ada != 0))
TN_values = np.sum(np.logical_and(real == 0, ada == 0))
FP_values = np.sum(np.logical_and(real == 0, ada != 0))
FN_values = np.sum(np.logical_and(real != 0, ada == 0))
r2 = r2_simple(real, ada)
mcc = GeneralInteraction.MCC_easy(TP_values, TN_values, FP_values, FN_values)
fpr = fpr_simple(fp=FP_values, tn=TN_values)

TP_values_2 = np.sum(np.logical_and(real_2 != 0, ada_2 != 0))
TN_values_2 = np.sum(np.logical_and(real_2 == 0, ada_2 == 0))
FP_values_2 = np.sum(np.logical_and(real_2 == 0, ada_2 != 0))
FN_values_2 = np.sum(np.logical_and(real_2 != 0, ada_2 == 0))
r2_2 = r2_simple(real_2, ada_2)
mcc_2 = GeneralInteraction.MCC_easy(TP_values_2, TN_values_2, FP_values_2, FN_values_2)
fpr_2 = fpr_simple(fp=FP_values_2, tn=TN_values_2)

TP_values_3 = np.sum(np.logical_and(real_3 != 0, ada_3 != 0))
TN_values_3 = np.sum(np.logical_and(real_3 == 0, ada_3 == 0))
FP_values_3 = np.sum(np.logical_and(real_3 == 0, ada_3 != 0))
FN_values_3 = np.sum(np.logical_and(real_3 != 0, ada_3 == 0))
r2_3 = r2_simple(real_3, ada_3)
mcc_3 = GeneralInteraction.MCC_easy(TP_values_3, TN_values_3, FP_values_3, FN_values_3)
fpr_3 = fpr_simple(fp=FP_values_3, tn=TN_values_3)

plt.rcParams['font.family'] = 'Arial'

fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col', sharey=True, figsize=(12, 12), dpi=100)

add_identity(ax, color='k', ls='--')
demo_2_0_copy, ada_2_0_copy = shrink_two_arr(real_2, ada_2)
ax.scatter(demo_2_0_copy, ada_2_0_copy, c='tab:red', s=100, marker='o', alpha=0.8, edgecolors='none')
ax.hlines(y=0, xmin=-10, xmax=100, color='k', linestyle='--')
ax.vlines(x=0, ymin=-10, ymax=100, color='k', linestyle='--')
ax.set_xlim(-0.1, 0.6)
ax.set_ylim(-0.1, 0.6)
ax.set_xlabel("True coupling", fontsize=24)
ax.set_ylabel("Inferred Coupling", fontsize=24)
ax.set_title(r"Pairwise coupling", fontsize=32)
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=24)
# ax.legend(labels=["Real", f"Adaptive LASSO, $R^2 = {round(r2_2, 4)}$, "
#                           f"$MCC = {round(mcc_2, 4)}$"], loc="lower right")
ax.legend(labels=["Real", f"Adaptive LASSO, $R^2 = {r2_2}$, "
                          f"$MCC = {round(mcc_2, 4)}$, "
                          f"$FPR_2 = {round(fpr_2, 8)}$"], loc="lower right")

path_eps = base_path + "fig_4_type_2.eps"
# fig.savefig(path_eps, format="eps", bbox_inches=None)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col', sharey=True, figsize=(12, 12), dpi=100)

add_identity(ax, color='k', ls='--')
demo_3_0_copy, ada_3_0_copy = shrink_two_arr(real_3, ada_3)
ax.scatter(demo_3_0_copy, ada_3_0_copy, c='tab:red', s=100, marker='^', alpha=0.8, edgecolors='none')
ax.hlines(y=0, xmin=-10, xmax=100, color='k', linestyle='--')
ax.vlines(x=0, ymin=-10, ymax=100, color='k', linestyle='--')
ax.set_xlim(-0.1, 0.6)
ax.set_ylim(-0.1, 0.6)
ax.set_xlabel("True coupling", fontsize=24)
ax.set_ylabel("Inferred Coupling", fontsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=24)
ax.set_title(r"Three-body coupling ", fontsize=32)
# ax.legend(labels=["Real", f"Adaptive LASSO, $R^2 = {round(r2_3, 4)}$, "
#                           f"$MCC = {round(mcc_3, 4)}$"], loc="lower right")
ax.legend(labels=["Real", f"Adaptive LASSO, $R^2 = {r2_3}$, "
                          f"$MCC = {round(mcc_3, 4)}$, "
                          f"$FPR_3 = {round(fpr_3, 8)}$"], loc="lower right")

path_eps = base_path + "fig_4_type_3.eps"
# fig.savefig(path_eps, format="eps", bbox_inches=None)
plt.show()
