import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler


def create_images_folder(x):
    if not os.path.exists(f'files/{x.__class__.__name__}/images/'):
        os.makedirs(f'files/{x.__class__.__name__}/images/')
    return f'files/{x.__class__.__name__}/images/'


file = uproot.open("files\PhaseSpaceSimulation.root")
events = file["PhaseSpaceTree"]
NumEntries = events.numentries  # number of data objects (vectors)
params = ["H1_PX", "H1_PY", "H1_PZ", "H2_PX", "H2_PY", "H2_PZ", "H3_PX", "H3_PY", "H3_PZ"]

Mass_K = 493.677


def sample_images(model, epoch, Average_mass_predicted, MPV_mass_predicted, sample_interval, P1x_data, P3z_data,
                  P_tot_data, E_tot_data, Mass_B_data, G_loss_epochs, D_loss_epochs, D_acc_epochs):
    r = 5000
    noise = np.random.normal(0, 1, (r, model.latent_dim))
    gen_raw = model.generator.predict(noise)
    gen_raw = gen_raw.reshape((gen_raw.shape[0], gen_raw.shape[1]))

    gen_p = model.scaler.inverse_transform(gen_raw)

    fig, axs = plt.subplots(3, 3)
    fig.set_size_inches(14, 14)
    Mass_B = np.zeros(r)
    P1x = np.zeros(r)
    P3z = np.zeros(r)
    P_tot = np.zeros(r)
    E_tot = np.zeros(r)
    for i in range(r):
        p_products = np.array([np.sqrt(np.square(gen_p[i][0]) + np.square(gen_p[i][1]) + np.square(gen_p[i][2])),
                               np.sqrt(np.square(gen_p[i][3]) + np.square(gen_p[i][4]) + np.square(gen_p[i][5])),
                               np.sqrt(np.square(gen_p[i][6]) + np.square(gen_p[i][7]) + np.square(gen_p[i][8]))])
        p_total = np.sqrt(np.square(gen_p[i][0] + gen_p[i][3] + gen_p[i][6]) +
                          np.square(gen_p[i][1] + gen_p[i][4] + gen_p[i][7]) +
                          np.square(gen_p[i][2] + gen_p[i][5] + gen_p[i][8]))
        E_total = np.sqrt(np.square(p_products) + Mass_K ** 2)
        Mass_B[i] = math.sqrt(np.sum(E_total) ** 2 - p_total ** 2)
        P1x[i] = gen_p[i][0]
        P3z[i] = gen_p[i][8]
        P_tot[i] = p_total
        E_tot[i] = np.sum(E_total)

    Average_mass_predicted.append(np.mean(Mass_B))

    n, bins, patches = axs[0, 0].hist(Mass_B, 200, range=(0, 50000), alpha=0.5, label='Generated data')
    # axs[0,0].hist(Mass_B_data[0:10000], 200, range=(0,50000), alpha=0.5, label = 'Input data')
    axs[0, 0].set_xlabel('Mass of the B meson [MeV]')
    axs[0, 0].set_ylabel('Number of counts')
    axs[0, 0].axvline(x=np.mean(Mass_B_data), color='r', linestyle='dashed')
    # axs[0,0].legend(loc='upper right')
    MPV_mass_predicted.append(np.mean(bins[np.where(n == np.amax(n))]))

    n2, bins2, patches2 = axs[0, 1].hist(P1x, 100, range=(-100000, 100000), alpha=0.5, label='Generated data')
    axs[0, 1].hist(P1x_data[0:r], 100, range=(-100000, 100000), label='Input data', alpha=0.5)
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].set_xlabel('Momentum X of K1 [MeV]')
    axs[0, 1].set_ylabel('Number of counts')

    axs[1, 0].plot(range(0, epoch + sample_interval, sample_interval), Average_mass_predicted, c='r', linewidth=4.0)
    axs[1, 0].set_xlim([0, 30000])
    axs[1, 0].set_ylim([100, 500000])
    axs[1, 0].set_xlabel('Epoch number')
    axs[1, 0].set_ylabel('Mean of the B mass predicted')
    axs[1, 0].plot(range(0, 30000 + sample_interval, 200), np.zeros(int(30000 / 200) + 1) + np.mean(Mass_B_data), 'm-.')
    axs[1, 0].set_yscale('log')

    axs[1, 1].plot(range(0, epoch + sample_interval, sample_interval), MPV_mass_predicted, c='g', linewidth=4.0)
    axs[1, 1].set_xlim([0, 30000])
    axs[1, 1].set_ylim([100, 500000])
    axs[1, 1].set_xlabel('Epoch number')
    axs[1, 1].set_ylabel('MPV of the B mass predicted')
    axs[1, 1].plot(range(0, 30000 + sample_interval, 200), np.zeros(int(30000 / 200) + 1) + np.mean(Mass_B_data), 'm-.')
    axs[1, 1].set_yscale('log')
    # fig.savefig("images/%d.png" % epoch)

    axs[2, 1].hist(P3z, 100, range=(0, 800000), alpha=0.5, label='Generated data')
    axs[2, 1].hist(P3z_data[0:r], 100, range=(0, 800000), label='Input data', alpha=0.5)
    axs[2, 1].legend(loc='upper right')
    axs[2, 1].set_xlabel('Momentum Z of K3 [MeV]')
    axs[2, 1].set_ylabel('Number of counts')

    axs[2, 0].plot(range(0, epoch + 1, 1), G_loss_epochs, c='g', linewidth=1.0, label='G loss')
    axs[2, 0].plot(range(0, epoch + 1, 1), D_loss_epochs, c='r', linewidth=1.0, label='D loss')
    axs[2, 0].plot(range(0, epoch + 1, 1), D_acc_epochs, c='c', linewidth=1.0, label='D accuracy')
    axs[2, 0].set_xlim([0, 30000])
    axs[2, 0].set_ylim([0, 2])
    axs[2, 0].set_xlabel('Epoch number')
    axs[2, 0].set_ylabel('Relative ratio')

    axs[0, 2].hist(P_tot, 100, range=(0, 1000000), alpha=0.5, label='Generated data')
    axs[0, 2].hist(P_tot_data[0:r], 100, range=(0, 1000000), label='Input data', alpha=0.5)
    axs[0, 2].legend(loc='upper right')
    axs[0, 2].set_xlabel('Total momentum [MeV]')
    axs[0, 2].set_ylabel('Number of counts')

    axs[1, 2].hist(E_tot, 100, range=(0, 1000000), alpha=0.5, label='Generated data')
    axs[1, 2].hist(E_tot_data[0:r], 100, range=(0, 1000000), label='Input data', alpha=0.5)
    axs[1, 2].legend(loc='upper right')
    axs[1, 2].set_xlabel('Total energy [MeV]')
    axs[1, 2].set_ylabel('Number of counts')
    fig.savefig(model.path + f'{epoch}.png')

    plt.close()


def prepare_data():
    data = events.lazyarrays(params)
    data_arr = np.vstack(list(data[elem] for elem in params)).T
    ped_calk = list()
    for i in range(NumEntries):
        ped_calk.append(np.sqrt(np.square(data_arr[i][0] + data_arr[i][3] + data_arr[i][6]) +
                                np.square(data_arr[i][1] + data_arr[i][4] + data_arr[i][7]) +
                                np.square(data_arr[i][2] + data_arr[i][5] + data_arr[i][8])))
    data_arr = np.hstack((data_arr, np.array(ped_calk)[:, None]))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 8)
    Mass_B = np.zeros(NumEntries)
    P1x = np.zeros(NumEntries)
    P3z = np.zeros(NumEntries)
    P_tot = np.zeros(NumEntries)
    E_tot = np.zeros(NumEntries)
    for i in range(NumEntries):
        p_products = np.array(
            [np.sqrt(np.square(data_arr[i][0]) + np.square(data_arr[i][1]) + np.square(data_arr[i][2])),
             np.sqrt(np.square(data_arr[i][3]) + np.square(data_arr[i][4]) + np.square(data_arr[i][5])),
             np.sqrt(np.square(data_arr[i][6]) + np.square(data_arr[i][7]) + np.square(data_arr[i][8]))])
        p_total = np.sqrt(np.square(data_arr[i][0] + data_arr[i][3] + data_arr[i][6]) +
                          np.square(data_arr[i][1] + data_arr[i][4] + data_arr[i][7]) +
                          np.square(data_arr[i][2] + data_arr[i][5] + data_arr[i][8]))
        E_total = np.sqrt(np.square(p_products) + Mass_K ** 2)
        Mass_B[i] = math.sqrt(np.sum(E_total) ** 2 - p_total ** 2)
        P1x[i] = data_arr[i][0]
        P3z[i] = data_arr[i][8]
        P_tot[i] = p_total
        E_tot[i] = np.sum(E_total)

    # Rescale -1 to 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data_arr)
    scaled_data = scaler.transform(data_arr)
    return scaler, scaled_data, P1x, P3z, P_tot, E_tot, Mass_B
