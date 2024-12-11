import numpy as np
import matplotlib.pyplot as plt

def add_noise(signal,H,n):
    noised_signals = H@signal + n
    return noised_signals
def create_signal_class(D, var_m, var_l, mean_m, mean_l, N):
    K = np.random.randint(1, D // 2 + 1, N)
    M = np.random.normal(mean_m, np.sqrt(var_m),N)
    L = np.random.normal(mean_l, np.sqrt(var_l),N)
    signal_autocorrelation = np.zeros((D, D))
    expect_signal = np.zeros((D,1))
    signals = []
    for i in range(N):
        signal = np.ones((D,1)) * M[i]
        signal[K[i]] += L[i]
        signal[D//2 - 1 + K[i]] += L[i]
        expect_signal += signal
        signal_autocorrelation += signal @ signal.T
        signals.append(signal)
    signals = np.hstack(signals)
    signal_autocorrelation /= N
    expect_signal /= N
    return signals, signal_autocorrelation, expect_signal

def plot_vector(vector, name):
    plt.figure()
    plt.plot(vector)
    plt.title(name)
    plt.savefig(name+'.png')

def plot_n_samples(n,N, signals, noised_signals, denoised_signals, name):
    samples = np.random.randint(0, N, n)
    plt.figure()
    fig, axs = plt.subplots(n, 1)
    for i, sample in enumerate(samples):
        sample_signal = np.array(signals[:, sample])
        sample_noisy = np.array(noised_signals[:, sample])
        sample_denoised = np.array(denoised_signals[:, sample])
        orig_plot, = axs[i].plot(sample_signal, label="org")
        noisy_plot, = axs[i].plot(sample_noisy, label="noi")
        reconstructed_plot, = axs[i].plot(sample_denoised, label="rec")
        axs[i].legend(handles=[orig_plot, noisy_plot, reconstructed_plot])
        axs[i].set_title("comparing the original, noisy and reconstructed signal " + str(sample))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(name, fontsize =16)
    plt.savefig(name + '.png')
    plt.show()
def show_matrix(mat, name):
    plt.figure()
    plt.imshow(mat, cmap='hot', interpolation='nearest')
    plt.title(name)
    plt.colorbar()
    plt.savefig(name + '.png')
    plt.show()

D = 64  # signal dimensions
c = 0.6  # param
N = 1000 # samples

expect_M, expect_L = 0, 0
var_M, var_L = c, (D*(1-c)/2)

signals, autocorrelation, expect_signal = create_signal_class(D, var_M, var_L, expect_M, expect_L, N)

# Plot empiric mean
plot_vector(expect_signal, 'Empiric Mean')

# Centrize autocorrelation matrix
autocorrelation -= expect_signal @ np.conjugate(expect_signal).T

# Show autocorrelation matrix
show_matrix(autocorrelation,'Empirical Autocorrelation Matrix')

# Create H's for each demo H[0] = I, H[1] = circulant, H[2] = circulant
H = [np.eye(D)]
H_first_row = np.zeros(D)
H_first_row[: 3] = [-5/2, 4/3, -1/12]
H_first_row[-2:] = [-1/12, 4/3]
H.append(np.array([[H_first_row[(j-i)%D] for j in range(D)] for i in range(D)]))

# Create n's params for each demo var_n[0] = 1, var_n[1] = 1, var_n[2] = 5
var_n = [(1,0),(5,0)]

for i,linear_operator in enumerate(H):
    for var, mean in var_n:
        # Apply noise to signals
        noised_signals = add_noise(signals,linear_operator, np.random.normal(mean, np.sqrt(var), (D,N)))

        # Construct filter
        weiner_filter = autocorrelation @ np.conjugate(linear_operator).T @ np.linalg.inv(
        linear_operator @ autocorrelation @ np.conjugate(linear_operator).T + (np.eye(D) * var))

        # Show filter
        show_matrix(weiner_filter,"Weiner_Filter_li_op_"+ str(i)+"_var_"+ str(var))

        # Apply filter
        denoised_signals = weiner_filter @ noised_signals

        # Plot sample results
        plot_n_samples(3,N, signals, noised_signals, denoised_signals, "Samples_li_op_"+ str(i)+"_var_"+ str(var))

        # Calculate empiric MSE's
        denoised_mse,  noised_mse = np.sum((signals - denoised_signals)**2)/(N*D), np.sum((signals - noised_signals)**2)/(N*D)
        print("li_op:"+ str(i)+" and var: "+ str(var)+" MSE's Are: ")
        print('Noised MSE: ', noised_mse)
        print('Denoised MSE: ', denoised_mse)

H = H[1]
H_pseudoinv = np.linalg.pinv(H)
show_matrix(H_pseudoinv, 'H - Pseudo-Inverse')
show_matrix(H_pseudoinv @ H, 'H multiply by Pseudo-Inverse')
show_matrix(H @ H_pseudoinv, 'Pseudo-Inverse multiply by H')
U,S,V = np.linalg.svd(H_pseudoinv)
zero_index = np.where(np.isclose(S, 0, atol=1e-10))[0]
phi1 = np.array([V[zero_index[0],:]]).T
phi2 = phi1.copy()*500
pinv_H_phi1 = H_pseudoinv @ phi1
pinv_H_phi2 = H_pseudoinv @ phi2
# Plot empiric mean
plot_vector(phi1, 'phi1')
plot_vector(phi2, 'phi2')
plot_vector((pinv_H_phi1), 'phi1 multiply by Pseudo-Inverse')
plot_vector((pinv_H_phi2), 'phi2 multiply by Pseudo-Inverse')
phi_l2 = np.sqrt(np.sum((phi1 - phi2)**2))
H_phi_l2 = np.sqrt(np.sum((pinv_H_phi1 - pinv_H_phi2)**2))
print('Phi(1-2) l2 norm is: ', phi_l2)
print('pinv_H Phi(1-2) l2 norm is: ', H_phi_l2)
