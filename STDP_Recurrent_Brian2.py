from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Simulation and Network Parameters
# -------------------------------
# Neuron numbers
N_E = 400
N_I = 80
N = N_E + N_I

# Model parameters (example values)
tau_m = 20*ms
V_rest = -70*mV
V_reset = -60*mV
V_thr = -50*mV
R_m = 10*Mohm

# STDP parameters (for excitatory-to-excitatory synapses)
tau_pre = 20*ms
tau_post = 20*ms
A_plus = 0.01
A_minus = 0.012
w_max = 1.0

# Simulation schedule
simulation_duration = 10*second  # run simulation for 10 seconds
defaultclock.dt = 0.5*ms

# Weight binarization threshold parameters for motif analysis:
w_distr_prctile = 99.  # 99th percentile

# -------------------------------
# Define Neuron Model Equations
# -------------------------------
eqs = '''
dv/dt = (V_rest - v + R_m*I_syn + R_m*I_drive)/tau_m : volt (unless refractory)
I_syn : amp
I_drive : amp
'''

# -------------------------------
# Create excitatory and inhibitory groups:
# -------------------------------
G_E = NeuronGroup(N_E, eqs, threshold='v>V_thr', reset='v=V_reset',
                  refractory=5*ms, method='exact')
G_I = NeuronGroup(N_I, eqs, threshold='v>V_thr', reset='v=V_reset',
                  refractory=2*ms, method='exact')
G_E.v = V_rest
G_I.v = V_rest

# PROVIDE A STRONGER CONSTANT DRIVE:
G_E.I_drive = 2*nA
G_I.I_drive = 2*nA

# Additionally, add external Poisson input to G_E to further drive activity:
P_input = PoissonGroup(N_E, rates=10*Hz)
S_input = Synapses(P_input, G_E, on_pre='v_post += 3*mV')
S_input.connect(j='i')

# -------------------------------
# Synapses: Excitatory-->Excitatory synapses with STDP
# -------------------------------
S_EE = Synapses(G_E, G_E,
                '''
                w : 1
                dpre/dt = -pre/tau_pre : 1 (event-driven)
                dpost/dt = -post/tau_post : 1 (event-driven)
                ''',
                on_pre='''
                v_post += w*mV
                pre = 1.
                w = clip(w + A_plus*post, 0, w_max)
                ''',
                on_post='''
                post = 1.
                w = clip(w - A_minus*pre, 0, w_max)
                ''',
                method='exact')
S_EE.connect(p=0.1)  # 10% connection probability
S_EE.w = 'rand()*w_max'

# -------------------------------
# Monitors
# -------------------------------
spike_mon_E = SpikeMonitor(G_E)
state_mon_E = StateMonitor(G_E, 'v', record=True)

# -------------------------------
# Run the Simulation
# -------------------------------
run(simulation_duration)

# -------------------------------
# Analysis Phase (*post*-simulation)
# -------------------------------
# (a) Raster plot of excitatory neurons:
plt.figure(figsize=(8,4))
plt.plot(spike_mon_E.t/ms, spike_mon_E.i, '.k', markersize=2)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index (E)')
plt.title('Raster Plot (Excitatory neurons)')
plt.show()

# (b) Compute instantaneous population firing rate for excitatory neurons
rate_bins = np.arange(0, simulation_duration/ms, 10)  # 10 ms bins
rate_counts, _ = np.histogram(spike_mon_E.t/ms, bins=rate_bins)
ifr_E = rate_counts / (len(G_E)*0.01)  # Hz
plt.figure(figsize=(8,4))
plt.plot(rate_bins[:-1], ifr_E, 'b-')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Population Firing Rate (E)')
plt.show()

# (c) Extract the weight matrix for EE connections.
W_EE = np.zeros((N_E, N_E))
for i, j, w in zip(S_EE.i, S_EE.j, S_EE.w):
    W_EE[i, j] = w

# (d) Motif analysis:
nonzero_weights = W_EE[W_EE > 0]
thr_wEE = np.percentile(nonzero_weights, w_distr_prctile)
B = (W_EE >= thr_wEE).astype(int)

def randomize_graph_2(B, n_iter):
    B_rand = B.copy()
    flat = B_rand.flatten()
    np.random.shuffle(flat)
    return flat.reshape(B_rand.shape)

B_rand = randomize_graph_2(B, int(1e4))

# (e) Compute matrix powers for motifs (length-2 paths)
b2 = np.linalg.matrix_power(B, 2)
b2r = np.linalg.matrix_power(B_rand, 2)
mask = (b2 > 0) & (b2r > 0)
b2_vals = b2[mask]
b2r_vals = b2r[mask]

plt.figure(figsize=(8,4))
plt.hist(b2_vals, bins=50, alpha=0.7, label='STDP Graph')
plt.hist(b2r_vals, bins=50, alpha=0.7, label='Randomized')
plt.xlabel('k')
plt.ylabel('Count')
plt.title('Histogram of LENGTH-2 k-parallel motifs')
plt.legend()
plt.show()

# (f) 2D histogram of in- vs. out-degrees (from binarized weights)
w_EE_discrete = (W_EE >= thr_wEE).astype(int)
num_incoming = np.sum(w_EE_discrete, axis=1)
num_outgoing = np.sum(w_EE_discrete, axis=0)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist2d(num_incoming, num_outgoing, bins=50, cmap='viridis')
plt.xlabel('Incoming edges (mean = %.2f)' % np.mean(num_incoming))
plt.ylabel('Outgoing edges (mean = %.2f)' % np.mean(num_outgoing))
plt.title('2D Histogram of In vs. Out Degrees')
plt.colorbar()
plt.subplot(1,2,2)
plt.scatter(num_outgoing, num_incoming, c='purple', edgecolors='k')
plt.xlabel('Outgoing edges')
plt.ylabel('Incoming edges')
plt.title(f'{simulation_duration/second:.2f} sec')
plt.grid(True)
plt.tight_layout()
plt.show()

# (g) Display binarized graph and a spy plot.
def show_binarized_graph(W, thr):
    B = (W >= thr).astype(int)
    plt.figure(figsize=(5,5))
    plt.imshow(B, cmap='gray', interpolation='nearest')
    plt.title('Binarized E-to-E Weight Graph', fontsize=16)
    plt.xlabel('Neuron index')
    plt.ylabel('Neuron index')
    plt.colorbar()
    plt.show()

show_binarized_graph(W_EE, thr_wEE)
plt.figure(figsize=(5,5))
plt.spy(W_EE >= thr_wEE)
plt.title('Spy Plot of Binarized E-to-E Weights', fontsize=16)
plt.show()

# (h) Auto-correlogram for one neuron (custom function):
def compute_autocorrelogram(spike_times, bin_size=5*ms, window=100*ms):
    if len(spike_times) == 0:
        raise ValueError('No spikes detected.')
    spike_times = np.array(spike_times)
    bin_size_sec = bin_size/second
    window_sec = window/second
    lags = np.arange(-window_sec, window_sec+bin_size_sec, bin_size_sec)
    counts = np.zeros(len(lags)-1)
    for t in spike_times:
        diffs = spike_times - t
        diffs = diffs[diffs != 0]  # exclude zero lag
        hist, _ = np.histogram(diffs, bins=lags)
        counts += hist
    counts = counts / len(spike_times)
    return lags[:-1] + bin_size_sec/2, counts

spike_times_0 = spike_mon_E.t[spike_mon_E.i==0] / second
bin_centers, ac_values = compute_autocorrelogram(spike_times_0)
plt.figure(figsize=(8,4))
plt.plot(bin_centers*1000, ac_values, 'k-', linewidth=2)
plt.xlabel('Lag (ms)')
plt.ylabel('Conditional firing rate (Hz)')
plt.title('Auto-correlogram for neuron 0 (E)')
plt.grid(True)
plt.show()