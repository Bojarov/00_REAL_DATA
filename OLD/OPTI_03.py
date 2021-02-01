import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

data_path_freq = './data/frequencies.npy'

with open(data_path_freq, 'rb') as frequencies:
    freq_list = list(np.load(frequencies, allow_pickle=True))
# print(freq_list)

data_path_volt = './data/voltages.npy'
with open(data_path_volt, 'rb') as voltages:
    volt_list = list(np.load(voltages, allow_pickle=True))
print(np.shape(volt_list))

n_steps, n_det, n_f, _ = np.shape(volt_list)


def f_b_field(x, I, y):
    """
    y-component of the magnetic field of infinite stright wire.
    """
    return 0.0000002 * I * y / (y * y + x * x)


def fit_params(n_steps, n_f, volt_list, factor=89000):
    fitparams_mat = np.zeros((n_steps, n_f, 2))
    for i in range(n_steps):
        for j in range(n_f):
            x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
            y_data = np.array(volt_list)[i, :, j, 1] / factor
            popt, pcov = curve_fit(f_b_field, x_data, y_data)

            fitparams_mat[i, j, :] = popt
    return fitparams_mat


def plot_bare_signal(step, volt_list, freq_list, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(n_f):
        x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
        y_data = np.array(volt_list)[step, :, i, 1] / factor
        ax.plot(x_data, y_data, marker='x', linestyle='', label='f=' + str(freq_list[i]) + "Hz")

    ax.legend()


def plot_bare_signal_symmetry_f_z(volt_list, freq_list, n_win, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    z_vec = np.linspace(0, 1, n_steps)
    fig = plt.figure(figsize=(3, 2))

    ax1 = fig.add_subplot(2, 1, 1)

    y_sym_mat = (np.array(volt_list)[:, 0, :, 1] - np.array(volt_list)[:, -1, :, 1]) / factor
    for i in range(n_f):
        # x_data = np.array([-0.5,-0.16667,0.16667, 0.5])
        # y_data = np.array(volt_list)[step, : ,i, 1]/factor
        # y_sym = y_data[0]-y_data[-1]

        ax1.plot(z_vec, y_sym_mat[:, i], linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)

    ax1.legend()
    ax1.set_xlim(0.0, 1.0)

    ax2 = fig.add_subplot(2, 1, 2)
    for i in range(n_f):
        f_y_sym = pd.Series(y_sym_mat[:, i])
        # print(f_y_sym.rolling(10).sum())
        f_y_sym_roll = np.array(f_y_sym.rolling(n_win, center=True).sum()) / n_win
        ax2.plot(z_vec, f_y_sym_roll, linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)
    ax2.legend()
    ax2.set_xlim(0.0, 1.0)


def plot_bare_signal_symmetry_norm_f_z(volt_list, freq_list, fit_params_mat, n_win, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    z_vec = np.linspace(0, 1, n_steps)
    fig = plt.figure(figsize=(3, 2))

    ax1 = fig.add_subplot(2, 1, 1)

    y_sym_mat = (np.array(volt_list)[:, 0, :, 1] - np.array(volt_list)[:, -1, :, 1]) / (
                fit_params_mat[:, :, 0] * factor)

    # for i in range(n_steps):
    #    for j in range(n_f):
    #        x_data = np.array([-0.5,-0.16667,0.16667, 0.5])
    #        y_data = np.array(volt_list)[i,:,j,1]/factor
    #        popt, pcov = curve_fit(f_b_field, x_data, y_data)
    #        
    #        fitparams_mat[i,j,:] = popt
    # fitparams_mat[:,:,0]

    for i in range(n_f):
        # x_data = np.array([-0.5,-0.16667,0.16667, 0.5])
        # y_data = np.array(volt_list)[step, : ,i, 1]/factor
        # y_sym = y_data[0]-y_data[-1]

        ax1.plot(z_vec, y_sym_mat[:, i], linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)

    ax1.legend()
    ax1.set_xlim(0.0, 1.0)

    ax2 = fig.add_subplot(2, 1, 2)
    for i in range(n_f):
        f_y_sym = pd.Series(y_sym_mat[:, i])
        # print(f_y_sym.rolling(10).sum())
        f_y_sym_roll = np.array(f_y_sym.rolling(n_win, center=True).sum()) / n_win
        ax2.plot(z_vec, f_y_sym_roll, linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)
    ax2.legend()
    ax2.set_xlim(0.0, 1.0)


def plot_bare_signal_and_fit(step, volt_list, freq_list, fit_params_mat, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)

    x_fit = np.linspace(-0.5, 0.5, 10000)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))

    for i in range(n_f):
        c = next(color)
        # bare data
        x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
        y_data = np.array(volt_list)[step, :, i, 1] / factor

        # fit data
        ax.plot(x_data, y_data, marker='x', linestyle='', c=c, label='f=' + str(freq_list[i]) + "Hz")

        popt = fit_params_mat[step, i, :]
        y_fit = f_b_field(x_fit, popt[0], popt[1])
        ax.plot(x_fit, y_fit, linestyle='-', c=c, label='f=' + str(freq_list[i]) + "Hz")

    ax.legend()


def plot_bare_signal_and_fit_norm(step, volt_list, freq_list, fit_params_mat, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)

    x_fit = np.linspace(-0.5, 0.5, 10000)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))

    for i in range(n_f):
        c = next(color)
        popt = fit_params_mat[step, i, :]

        # bare data normalized by fittet current
        x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
        y_data = np.array(volt_list)[step, :, i, 1] / factor

        # fit data

        # if i==0:

        ax.plot(x_data, y_data / popt[0], marker='x', linestyle='', c=c, label='f=' + str(freq_list[i]) + "Hz")

        y_fit = f_b_field(x_fit, popt[0], popt[1])
        ax.plot(x_fit, y_fit / popt[0], linestyle='-', c=c, label='f=' + str(freq_list[i]) + "Hz")

    ax.legend()


def plot_depth_f_z(fit_params_mat, freq_list):
    n_steps, n_f, _ = np.shape(fit_params_mat)

    z_vec = np.linspace(0, 1, n_steps)

    fig = plt.figure(figsize=(3, 2))

    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_f):
        ax.plot(z_vec, fit_params_mat[:, i, 1], label='f=' + str(freq_list[i]) + "Hz")
    ax.legend()


def plot_current_f_z(fit_params_mat, freq_list):
    n_steps, n_f, _ = np.shape(fit_params_mat)

    z_vec = np.linspace(0, 1, n_steps)
    fig = plt.figure(figsize=(3, 2))

    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_f):
        ax.plot(z_vec, fit_params_mat[:, i, 0], label='f=' + str(freq_list[i]) + "Hz")
    ax.legend()


fit_params_mat = fit_params(n_steps, n_f, volt_list)

# plot_bare_signal(0, volt_list, freq_list)

plot_bare_signal_and_fit(0, volt_list, freq_list, fit_params_mat)

plot_bare_signal_symmetry_f_z(volt_list, freq_list, n_win=20)

plot_bare_signal_symmetry_norm_f_z(volt_list, freq_list, fit_params_mat, n_win=20)

# plot_depth_f_z(fit_params_mat, freq_list)

plot_current_f_z(fit_params_mat, freq_list)

plt.show()
