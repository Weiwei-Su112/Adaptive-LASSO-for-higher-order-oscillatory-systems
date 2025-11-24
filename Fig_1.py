from GeneralInteraction import *


def make_dirs(conn_seed_):
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    tm_string = now.strftime("%H%M%S") + "_" + str(conn_seed_)
    file_path1 = "tmp/img/" + date + "/"
    file_path_time = "tmp/img/" + date + "/" + tm_string + "/"
    file_path2 = "tmp/img/" + date + "/" + tm_string + "/2/"
    file_path3 = "tmp/img/" + date + "/" + tm_string + "/3/"
    file_path_mix = "tmp/img/" + date + "/" + tm_string + "/mix/"
    os.makedirs(file_path1, exist_ok=True)
    os.mkdir(file_path_time)
    os.mkdir(file_path2)
    os.mkdir(file_path3)
    os.mkdir(file_path_mix)
    return file_path_time, file_path2, file_path3, file_path_mix


def make_mp4(path, int_type, fps_, movie_path):
    if int_type == 2:
        os.system(f"ffmpeg -f image2 -r {str(fps_)} -i {path}nb%d.jpeg -vcodec mpeg4 -y "
                  f"{movie_path}movie_pairwise.mp4")
    elif int_type == 3:
        os.system(f"ffmpeg -f image2 -r {str(fps_)} -i {path}nb%d.jpeg -vcodec mpeg4 -y "
                  f"{movie_path}movie_3-interaction.mp4")
    elif int_type == 0:
        os.system(f"ffmpeg -f image2 -r {str(fps_)} -i {path}nb%d.jpeg -vcodec mpeg4 -y "
                  f"{movie_path}movie_mixture.mp4")
    return None


def special_eps(acts, plts, what_time, type):
    difference_array = np.absolute(plts - what_time)
    index = difference_array.argmin()
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5),
                           subplot_kw={
                               "ylim": (-1.1, 1.1),
                               "xlim": (-1.1, 1.1),
                               "xlabel": r'$\cos(\theta)$',
                               "ylabel": r'$\sin(\theta)$',
                           })
    ax.plot(np.cos(acts[:, index]),
            np.sin(acts[:, index]),
            'o',
            markersize=10, color="blue")
    r1 = round(GeneralInteraction.phase_coherence(acts.T[index]), 3)
    if type == 2:
        ax.set_title(f'Time = {round(plts[index], 1)}, $R_1$ = {r1}, Pairwise')
        save_path = path_total + "ss_2_" + str(what_time) + ".eps"
    if type == 3:
        ax.set_title(f'Time = {round(plts[index], 1)}, $R_1$ = {r1}, 3-interaction')
        save_path = path_total + "ss_3_" + str(what_time) + ".eps"
    if type == 5:
        ax.set_title(f'Time = {round(plts[index], 1)}, $R_1$ = {r1}, Mixture')
        save_path = path_total + "ss_mix_" + str(what_time) + ".eps"
    plt.savefig(save_path, format="eps", bbox_inches=None)
    plt.close('all')


T = 900
dt = 0.02
p = 0.1
starts_from = 1 / 9
cut_time = int(T * starts_from)
times = np.arange(0, T - cut_time, 2 * np.pi / 50)

conn_seed = 112
noise_seed = 3999
noise_seed2, noise_seed3, noise_seedmix = np.random.default_rng(noise_seed).integers(0, 1e10, size=3)
reduce_seed = 10001

take_slice = 0

save = False
video = False
fps = 50

need_sps = [0.0, 0.0, 0.0]
need_start = 0.0
# need_end = T * (1 - starts_from) - max(need_sps)
need_end = 600.0
inf_last = (need_start + need_end) / T
index_len = int((need_start + need_end) / dt)

if conn_seed == 112:
    v_mixes = [[125.2], [325.0], [127.7]]  # [[125.2], [325.0], [16.6]] for conn_seed = 112
else:
    v_mixes = [[100.0], [100.0], [100.0]]  # [[125.2], [325.0], [16.6]] for conn_seed = 112
plt_x_axis_2 = np.arange(0, need_end + need_start, dt)[:index_len]
plt_x_axis_3 = np.arange(0, need_end + need_start, dt)[:index_len]
plt_x_axis_mix = np.arange(0, need_end + need_start, dt)[:index_len]

plt_x_axis_lst = [plt_x_axis_2, plt_x_axis_3, plt_x_axis_mix]
# for noise3489 ==========================================

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']

if save:
    path_total, path_2, path_3, path_mix = make_dirs(conn_seed_=conn_seed)
# ==============================================================================
cou2 = 0.1
cou3 = 0.6

# # ----------------------------------------------
some_rng = np.random.default_rng(9876)
natfreq = some_rng.normal(loc=1, scale=0.1, size=12)
model_2 = GeneralInteraction(coupling2=cou2, coupling3=0, dt=dt, T=T, natfreqs=natfreq, with_noise=True,
                             noise_sth=0.2, normalize=True, conn2=p, all_connected=False,
                             conn_seed=conn_seed, noise_seed=noise_seed2, starts_from=starts_from,
                             inf_last=inf_last, old_legacy=True)
test_conn2 = model_2.conn_mat2
test_coup2 = model_2.coupling2
init_phase = model_2.init_phase
act_mat2 = model_2.run()

__, ops2 = plot_phase_coherence(act_mat2, color="blue", coup2=cou2, x_axis=plt_x_axis_2, v=v_mixes[0],
                                take_slice=take_slice)
if save:
    save_path_2 = path_total + "pairwise.eps"
    plt.savefig(save_path_2, format="eps", bbox_inches=None)
plt.show()
plt.close("all")

for i in range(len(times)):
    difference_array = np.absolute(plt_x_axis_2 - times[i])
    index = difference_array.argmin()
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5),
                           subplot_kw={
                               "ylim": (-1.1, 1.1),
                               "xlim": (-1.1, 1.1),
                               "xlabel": r'$\cos(\theta)$',
                               "ylabel": r'$\sin(\theta)$',
                           })
    phases = act_mat2[:, index]
    nodes = np.arange(start=1, stop=len(phases) + 1, dtype=np.int32)
    coss = np.cos(phases)
    sins = np.sin(phases)
    write_lst2_1 = np.array([nodes, phases, coss, sins]).T.tolist()
    ax.plot(np.cos(act_mat2[:, index]),
            np.sin(act_mat2[:, index]),
            'o',
            markersize=10, color="blue")
    r1 = round(GeneralInteraction.phase_coherence(act_mat2.T[index]), 3)
    ax.set_title(f'Time = {round(plt_x_axis_2[index], 1)}, $R_1$ = {r1}, Pairwise')
    # plt.show()
    if save:
        # if round(plt_x_axis_2[index], 2) == v_mixes[0][0]:
        #     save_path_2 = path_total + "ss_2_" + str(v_mixes[0][0]) + ".eps"
        #     plt.savefig(save_path_2, format="eps", bbox_inches=None)
        path2 = path_2 + "nb" + str(i + 1) + ".jpeg"
        plt.savefig(path2)
    plt.close('all')

if save and video:
    special_eps(act_mat2, plt_x_axis_2, v_mixes[0][0], 2)
    make_mp4(path_2, 2, fps, movie_path=path_total)

# ----------------------------------------------
model_3 = GeneralInteraction(coupling2=0, coupling3=cou3, dt=dt, T=T, natfreqs=natfreq, with_noise=True,
                             noise_sth=0.2, normalize=True, conn3=p, all_connected=False, init_phase=init_phase,
                             conn_seed=conn_seed, noise_seed=noise_seed3, starts_from=starts_from,
                             inf_last=inf_last, old_legacy=True)
act_mat3 = model_3.run()
test_conn3 = model_3.conn_mat3
test_coup3 = model_3.coupling3

model_3.init_phase = init_phase
act_mat3 = model_3.run()

__, ops3 = plot_phase_coherence(act_mat3, color="orange",
                                coup3=cou3, x_axis=plt_x_axis_3, v=v_mixes[1], take_slice=take_slice)
if save:
    save_path_3 = path_total + "3-interaction.eps"
    plt.savefig(save_path_3, format="eps", bbox_inches=None)
plt.show()
plt.close("all")

for i in range(len(times)):
    difference_array = np.absolute(plt_x_axis_3 - times[i])
    index = difference_array.argmin()
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5),
                           subplot_kw={
                               "ylim": (-1.1, 1.1),
                               "xlim": (-1.1, 1.1),
                               "xlabel": r'$\cos(\theta)$',
                               "ylabel": r'$\sin(\theta)$',
                           })
    phases = act_mat3[:, index]
    nodes = np.arange(start=1, stop=len(phases) + 1, dtype=np.int32)
    coss = np.cos(phases)
    sins = np.sin(phases)
    write_lst2_1 = np.array([nodes, phases, coss, sins]).T.tolist()
    ax.plot(np.cos(act_mat3[:, index]),
            np.sin(act_mat3[:, index]),
            'o',
            markersize=10, color="orange")
    r1 = round(GeneralInteraction.phase_coherence(act_mat3.T[index]), 3)
    ax.set_title(f'Time = {round(plt_x_axis_3[index], 1)}, $R_1$ = {r1}, 3-interaction')
    if save:
        if round(plt_x_axis_3[index], 2) == v_mixes[1][0]:
            save_path_3 = path_total + "ss_3_" + str(v_mixes[1][0]) + ".eps"
            plt.savefig(save_path_3, format="eps", bbox_inches=None)
        path3 = path_3 + "nb" + str(i + 1) + ".jpeg"
        plt.savefig(path3)
    plt.close("all")

if save and video:
    special_eps(act_mat3, plt_x_axis_3, v_mixes[1][0], 3)
    make_mp4(path_3, 3, fps, movie_path=path_total)

# # ==============================================================================
test_conn_mix2 = reduce_conn_2(test_conn2, 0.5, reduce_seed)
test_coup_mix2 = test_conn_mix2 * cou2 / (p * 12)
test_conn_mix3 = reduce_conn_3(test_conn3, 0.5, reduce_seed)
test_coup_mix3 = test_conn_mix3 * cou3 / (p * 12 * 12)
model_mix = GeneralInteraction(dt=dt, T=T, natfreqs=natfreq, with_noise=True, pre_coup2=test_coup_mix2,
                               pre_coup3=test_coup_mix3, init_phase=init_phase,
                               pre_conn2=test_conn_mix2, pre_conn3=test_conn_mix3,
                               noise_sth=0.2, conn_seed=conn_seed, noise_seed=noise_seed3, starts_from=starts_from,
                               inf_last=inf_last, conn=p/2)
# model_mix.init_phase = init_phase
act_mat_mix = model_mix.run()

__, ops_mix = plot_phase_coherence(act_mat_mix, color="green",
                                   coup2=cou2, coup3=cou3,
                                   x_axis=plt_x_axis_mix, v=v_mixes[2], take_slice=take_slice)
if save:
    save_path_mix = path_total + "mixture.eps"
    plt.savefig(save_path_mix, format="eps", bbox_inches=None)
plt.show()
plt.close("all")

xticks = np.arange(0, need_start + need_end + 0.1, 100)
plot_phase_coherence_3set_all(act_mat2,
                              act_mat3,
                              act_mat_mix, x_axis=plt_x_axis_lst,
                              coup2=cou2, coup3=cou3, coup2_mix=cou2,
                              coup3_mix=cou3, v=v_mixes, xticks=xticks)
if save:
    save_path_all = path_total + "all.eps"
    plt.savefig(save_path_all, format="eps", bbox_inches=None)
plt.show()
plt.close("all")

for i in range(len(times)):
    difference_array = np.absolute(plt_x_axis_mix - times[i])
    index = difference_array.argmin()
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5),
                           subplot_kw={
                               "ylim": (-1.1, 1.1),
                               "xlim": (-1.1, 1.1),
                               "xlabel": r'$\cos(\theta)$',
                               "ylabel": r'$\sin(\theta)$',
                           })
    phases = act_mat_mix[:, index]
    nodes = np.arange(start=1, stop=len(phases) + 1, dtype=np.int32)
    coss = np.cos(phases)
    sins = np.sin(phases)
    write_lst2_1 = np.array([nodes, phases, coss, sins]).T.tolist()
    ax.plot(np.cos(act_mat_mix[:, index]),
            np.sin(act_mat_mix[:, index]),
            'o',
            markersize=10, color="green")
    r1 = round(GeneralInteraction.phase_coherence(act_mat_mix.T[index]), 3)
    ax.set_title(f'Time = {round(plt_x_axis_mix[index], 1)}, $R_1$ = {r1}, Mixture')
    if save:
        if round(plt_x_axis_mix[index], 2) == v_mixes[2][0]:
            save_path_mix = path_total + "ss_mix_" + str(v_mixes[2][0]) + ".eps"
            plt.savefig(save_path_mix, format="eps", bbox_inches=None)
        pathmix = path_mix + "nb" + str(i + 1) + ".jpeg"
        plt.savefig(pathmix)
    plt.close("all")

if save and video:
    special_eps(act_mat_mix, plt_x_axis_mix, v_mixes[2][0], 5)
    make_mp4(path_mix, 0, fps, movie_path=path_total)

# # ==============================================================================
Fig1_demo(model_2, "Pairwise")
Fig1_demo(model_3, "3-interaction")
Fig1_demo(model_mix, "Mixture")

plt.hist(ops2, bins=np.arange(0.0, 1.0, 0.1))
plt.title("Pairwise R")
plt.xticks(np.arange(0.0, 1.0, 0.1))
plt.show()
plt.hist(ops3, bins=np.arange(0.0, 1.0, 0.1))
plt.title("3-int R")
plt.xticks(np.arange(0.0, 1.0, 0.1))
plt.show()
plt.hist(ops_mix, bins=np.arange(0.0, 1.0, 0.1))
plt.title("Mix R")
plt.xticks(np.arange(0.0, 1.0, 0.1))
plt.show()
