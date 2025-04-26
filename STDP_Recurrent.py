import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
from scipy import sparse
import networkx as nx
import copy
import pickle
import os
import argparse

def motiv(W, C_EE_FFI, FFI_thr):
    """
    Counts small motifs in a directed graph.
    
    Parameters:
        W : numpy.ndarray
            The adjacency matrix of the graph (0's and 1's) with zeros on the diagonal.
        C_EE_FFI : numpy.ndarray
            A matrix used to apply the feed-forward inhibition threshold.
        FFI_thr : float
            The threshold value for moderate feed-forward inhibition.
    
    Returns:
        bpm : float
            Number of biparallel motifs: i --> k --> j and i --> l --> j with k ≠ l.
        bfm : float
            Number of bifan motifs: i --> k <-- j and i --> l <-- j with k ≠ l.
        oc : float
            Number of open chain motifs: i --> k --> j with i ≠ j.
        lo : float
            Number of loops: i --> j and j --> i.
    """
    # Get number of nodes.
    N = W.shape[0]
    
    # Create a mask that is 1 everywhere except on the diagonal.
    T1 = np.ones((N, N)) - np.eye(N)
    
    # Compute W2 which counts the number of 2-paths (j --> k --> i).
    # Then restrict to paths where C_EE_FFI is less than FFI_thr.
    W2 = W @ W
    W2 = W2 * (C_EE_FFI < FFI_thr)
    
    # Component-wise square of W2.
    W2cs = W2 ** 2
    
    # Biparallel motifs are given by (W2cs - W2)/2.
    BPM = (W2cs - W2) / 2
    # Sum all off-diagonal elements to avoid self motifs.
    bpm = np.sum(BPM * T1)
    
    # Compute WtW to count 2-paths of the form (j --> k <-- i).
    WtW = W.T @ W
    WtWcs = WtW ** 2
    
    # Bifan motifs are given by (WtWcs - WtW)/2.
    BFM = (WtWcs - WtW) / 2
    bfm = np.sum(BFM * T1)
    
    # Loops: mutual links between nodes (i --> j and j --> i) are captured by the diagonal of W2.
    lo = np.trace(W2) / 2
    
    # Open chain motifs: total number of 2-paths minus those that are loops.
    oc = np.sum(W2) - 2 * lo
    
    return bpm, bfm, oc, lo

def motiv3(W, C_EE_FFI, FFI_thr):
    """
    Counts biparallel motifs in a directed graph.
    
    Parameters:
      W        : {0,1}-valued adjacency matrix (either dense or sparse)
      C_EE_FFI : matrix for feed-forward inhibition (dense or sparse)
      FFI_thr  : threshold value (float)
    
    Returns:
      bpm : float
            Total count of biparallel motifs.
    """
    # Ensure W is a dense numpy array.
    if sparse.issparse(W):
        W = W.toarray()
    # Similarly, ensure C_EE_FFI is dense.
    if sparse.issparse(C_EE_FFI):
        C_EE_FFI = C_EE_FFI.toarray()

    N = W.shape[0]
    T1 = np.ones((N, N)) - np.eye(N)

    # Compute the number of 2-paths.
    W2 = W @ W

    # Create boolean mask based on C_EE_FFI and the threshold.
    mask = C_EE_FFI < FFI_thr
    W2 = W2 * mask  # Element-wise multiplication.

    # Compute the element-wise square.
    W2cs = W2 ** 2

    # Compute motif counts.
    BPM = (W2cs - W2) / 2
    bpm = np.sum(BPM * T1)
    return bpm

def motiv4(W):
    """
    Counts bifan motifs in a directed graph.

    The graph is represented by the adjacency matrix W (a binary {0,1}-valued NumPy array).
    Bifan motifs are defined as having two converging 2-paths (i → k ← j and i → l ← j with k ≠ l).
    Only off-diagonal contributions are considered.

    Parameters:
        W : numpy.ndarray
            The adjacency matrix of the graph.

    Returns:
        bfm : float
            The number of bifan motifs in the graph.
    """
    N = W.shape[0]
    # T1 is a mask with ones off the diagonal and zeros on the diagonal.
    T1 = np.ones((N, N)) - np.eye(N)

    # Compute WtW = W.T @ W, where WtW[i,j] counts the number of paths of the form j → k ← i.
    WtW = W.T @ W
    # Component-wise square of WtW.
    WtWcs = WtW ** 2
    # Bifan motif matrix.
    BFM = (WtWcs - WtW) / 2
    # Sum only off-diagonal entries.
    bfm = np.sum(BFM * T1)
    return bfm

def motivity_coef_abs(W):
    """
    Compute the absolute (as opposed to within-support) motivity coefficient 
    for a graph represented by an adjacency matrix W.
    
    Parameters:
        W : numpy.ndarray
            A square [0,1]- or {0,1}-valued matrix representing the graph.
            It is assumed that W is of shape (N, N).
            
    Returns:
        Mot_coef_abs : float
            The absolute motivity coefficient.
    """
    # Number of nodes
    N = W.shape[0]
    
    # T1 is a matrix of ones except zeros on the diagonal.
    T1 = np.ones((N, N)) - np.eye(N)
    
    # Compute WW = W @ W (matrix multiplication), then zero out the diagonal.
    WW = W @ W
    WW = T1 * WW  # elementwise multiplication to set the diagonal to 0
    
    # WoW is the elementwise square of W.
    WoW = W * W  # for binary matrices, this equals W, but works for non-binary values too.
    
    # Compute the sum of WoW along rows to get a 1D array of column sums.
    col_sum_W2 = np.sum(WoW, axis=0)           # Shape: (N,)
    # Create an (N, N) matrix where each row i is col_sum_W2[i]
    col_sum_W2_r = np.tile(col_sum_W2.reshape(N, 1), (1, N))
    
    # Compute the sum of WoW along columns to get a 1D array of row sums.
    row_sum_W2 = np.sum(WoW, axis=1)           # Shape: (N,)
    # Create an (N, N) matrix where every row is row_sum_W2 (as a row vector)
    row_sum_W2_r = np.tile(row_sum_W2.reshape(1, N), (N, 1))
    
    # Compute transposes and intermediate matrices.
    Wt = W.T
    WtWW = Wt @ WW      # Counts contribution from paths (not Hadamard-multiplied by support)
    WWWt = WW @ Wt
    
    # Compute adjustments to subtract contributions from completed square terms.
    # Note that WoW.T is the transpose of the elementwise square of W.
    W_csW2 = W * (col_sum_W2_r - WoW.T)
    W_rsW2 = W * (row_sum_W2_r - WoW.T)
    
    # Compute the numerator and denominator for the coefficient.
    numerator = np.sum(W * (WtWW + WWWt - W_csW2 - W_rsW2))
    denominator = np.sum(WtWW + WWWt - W_csW2 - W_rsW2)
    
    # Avoid division by zero.
    if denominator != 0:
        Mot_coef_abs = numerator / denominator
    else:
        Mot_coef_abs = np.nan  # or set to zero, depending on desired behavior
    
    return Mot_coef_abs

def randomize_graph_2(B, n_ite):
    """
    Takes in B, an N×N {0,1} matrix with zeros on the diagonal,
    i.e. B is the adjacency matrix of a loop-free directed graph.
    Returns a randomized matrix C with exactly the same combined IN- and OUT-degree sequence
    and with zeros on the diagonal.
    
    Parameters:
      B: numpy.ndarray or a sparse matrix representing an N×N adjacency matrix.
      n_ite: number of iterations to perform randomization.
    
    Returns:
      C: a sparse matrix (CSR format) with the randomized structure.
    """
    # If B is sparse, convert to a dense numpy array.
    if hasattr(B, "toarray"):
        B = B.toarray()
    B = np.array(B)  # ensure B is a NumPy array
    N = B.shape[0]
    C = B.copy()
    
    for _ in range(n_ite):
        # --- Row swapping ---
        # pick two distinct row indices
        r1 = np.random.randint(N)
        # choose a random integer in [0, N-1) excluding r1:
        candidate = np.random.randint(N - 1)
        r2 = candidate if candidate < r1 else candidate + 1

        # Find the column indices where the two rows differ.
        ndx = np.where(C[r1, :] != C[r2, :])[0]
        if ndx.size > 0:
            # Randomly permute the mismatch column indices.
            perm = np.random.permutation(len(ndx))
            ndx2 = ndx[perm]
            # Get the submatrix corresponding to the two rows and permuted columns.
            submatrix = C[[r1, r2], :][:, ndx2]
            
            # Check if swapping would put a 1 on the diagonal.
            # (For the rows, the diagonal elements are C[r1, r1] and C[r2, r2].)
            idx_r1 = np.where(ndx == r1)[0]
            idx_r2 = np.where(ndx == r2)[0]
            cond1 = (idx_r1.size == 0) or (submatrix[0, idx_r1[0]] == 0)
            cond2 = (idx_r2.size == 0) or (submatrix[1, idx_r2[0]] == 0)
            
            if cond1 and cond2:
                # Swap the entries along the mismatch columns.
                C[np.ix_([r1, r2], ndx)] = submatrix

        # --- Column swapping ---
        # pick two distinct column indices (using similar logic as for rows)
        c1 = np.random.randint(N)
        candidate = np.random.randint(N - 1)
        c2 = candidate if candidate < c1 else candidate + 1

        # Find the row indices where the two columns differ.
        ndx = np.where(C[:, c1] != C[:, c2])[0]
        if ndx.size > 0:
            # Randomly permute the mismatch row indices.
            perm = np.random.permutation(len(ndx))
            ndx2 = ndx[perm]
            # Get the submatrix corresponding to the permuted rows and the two columns.
            submatrix = C[ndx2, :][:, [c1, c2]]
            
            # Check if swapping would put a 1 on the diagonal.
            # For columns, the diagonal elements that might be affected are C[c1, c1] and C[c2, c2].
            idx_c1 = np.where(ndx == c1)[0]
            idx_c2 = np.where(ndx == c2)[0]
            cond1 = (idx_c1.size == 0) or (submatrix[idx_c1[0], 0] == 0)
            cond2 = (idx_c2.size == 0) or (submatrix[idx_c2[0], 1] == 0)
            
            if cond1 and cond2:
                C[np.ix_(ndx, [c1, c2])] = submatrix

    # Return the final matrix as a sparse matrix (CSR format)
    return sparse.csr_matrix(C)

def shuffle_the_matrix(A, W):
    """
    Shuffles the entries of the matrix W within the support defined by 
    the adjacency (or binary) matrix A.

    Parameters:
        A : ndarray
            A binary matrix (e.g., an adjacency matrix) defining the support
            where shuffling occurs.
        W : ndarray
            A matrix (of the same shape as A) whose values at positions where A is True
            will be shuffled.

    Returns:
        W_shuf : scipy.sparse.coo_matrix
            A sparse matrix (in COO format) with shape (N, N) that contains the shuffled
            values of W at the positions where A is True.
    """
    # Determine the size (assumes W is square)
    N = W.shape[0]

    # Convert A to a boolean mask (in case it is not already boolean)
    A_bool = A.astype(bool)

    # Extract the values from W corresponding to True entries in A
    W_val = W[A_bool]

    # Get the row and column indices where A is True
    rows, cols = np.nonzero(A_bool)
    
    # Number of edges / nonzero entries in the support
    nnn = len(rows)
    
    # Permute the indices at random
    shuf_edges = np.random.permutation(nnn)
    
    # Assign the shuffled values
    W_val_shuf = W_val[shuf_edges]
    
    # Create the sparse matrix using the shuffled values.
    W_shuf = sparse.coo_matrix((W_val_shuf, (rows, cols)), shape=(N, N))
    
    return W_shuf

def binarized_graph(W_ee, threshold):
    """
    Produces a binarized directed graph based on a threshold.
    
    Parameters:
        W_ee (numpy.ndarray): The weighted adjacency matrix of the graph.
        threshold (float): The threshold value used to binarize W_ee.
    
    Returns:
       plt figure with binarized directed graph
        
    The function:
      - Removes diagonal (self-loop) contributions,
      - Binarizes the matrix (B = W_ee >= threshold),
      - Constructs a directed graph from B using networkx,
      - Plots the graph in 3D with a force-directed layout,
      - Computes the absolute motivity coefficient and that
        of a shuffled version of the binarized matrix.
    """
    
    # Number of nodes
    N_E = W_ee.shape[0]
    
    # Create a mask to remove diagonal elements (self-loops).
    T1 = np.ones((N_E, N_E)) - np.eye(N_E)
    
    # Binarize the graph.
    B = (W_ee >= threshold).astype(int)
    
    # Create a directed graph using the binarized matrix.
    # (nx.from_numpy_array interprets the array as an adjacency matrix.)
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)
    N_edges = G.number_of_edges()
    
    # Compute a 3D spring (force-directed) layout.
    pos = nx.spring_layout(G, dim=3)
    
    # Set up a 3D plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract node positions.
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    
    # Draw nodes with marker size ~4 (adjusted via 's') and green color.
    ax.scatter(xs, ys, zs, s=20, c='g')
    
    # Draw edges as lines (the arrow heads are not drawn).
    for u, v in G.edges():
        x_coords = [pos[u][0], pos[v][0]]
        y_coords = [pos[u][1], pos[v][1]]
        z_coords = [pos[u][2], pos[v][2]]
        ax.plot(x_coords, y_coords, z_coords, color='black', linewidth=1)
    
    # Compute the motivity coefficient for the binarized matrix.
    mot_coef = motivity_coef_abs(B)
    # Shuffle the matrix while retaining support structure.
    B_shuf = shuffle_the_matrix(T1, B)
    mot_coef_shuf = motivity_coef_abs(B_shuf)
    
    # Create a title with threshold, number of edges, and the motivity coefficients.
    title_text = (f"θ = {threshold:.2f}, {N_edges} edges,\n"
                  f"Abs-c_M:  W={mot_coef:.2f}  Shuf={mot_coef_shuf:.2f}")
    ax.set_title(title_text, fontsize=16)
    
    # Set the axis tick label font size.
    ax.tick_params(labelsize=12)
    
    return fig

def setup_stdp(N: int = 500, p_E: float = 0.8, connect_type: int = 2):
    """
    Set up the network, parameters, and initial conditions for the STDP simulation.

    Parameters:
    N: number of neurons
    p_E: fraction of excitatory cells
    connect_type: 1 for ER connectivity; 2 for homogeneous in-degrees
    
    Returns:
      setup_dict (dict): A dictionary containing network structure, simulation parameters,
        initial state variables, connectivity matrices, weight matrices, and motif-analysis
        variables.
    """

    # -------------------------
    # NETWORK SETUP 
    # -------------------------

    N_E = int(round(N * p_E))   # number of excitatory neurons
    N_I = N - N_E               # number of inhibitory neurons

    # We use 0-indexing in Python.
    ee = np.arange(N_E)         # indices of excitatory neurons
    ii = np.arange(N_E, N)       # indices of inhibitory neurons

    # ER parameter (if connect_type==1)
    p_ER = 0.7

    # Connection probabilities for homogeneous in-degrees:
    p_EE_val = 0.9      # E --> E
    p_IE = 0.5          # E --> I (for I neurons, presynaptic from E)
    p_EI = 0.5          # I --> E
    p_II = 0.5          # I --> I
    
    if connect_type == 1:
        # Using an Erdos-Renyi (ER) connectivity model.
        C = (np.random.rand(N, N) < p_ER).astype(float)
        np.fill_diagonal(C, 0)  # Exclude self-connections
        # Split connectivity into submatrices:
        C_EE = C[np.ix_(ee, ee)]
        C_IE = C[np.ix_(ii, ee)]
        C_EI = C[np.ix_(ee, ii)]
        C_II = C[np.ix_(ii, ii)]
    else:
        # Homogeneous in-degrees: for each postsynaptic neuron,
        # a fixed number of presynaptic neurons is randomly selected.
        # For postsynaptic excitatory neurons:
        N_EE = int(round(N_E * p_EE_val))   # number of excitatory presynaptic connections to an excitatory neuron
        N_EI = int(round(N_I * p_EI))         # number of inhibitory presynaptic connections to an excitatory neuron
        
        pre_EE_list, post_EE_list = [], []
        pre_EI_list, post_EI_list = [], []
        for j_post in range(N_E):
            # Exclude self-connection:
            available_E = np.delete(np.arange(N_E), j_post)
            j_pre_EE = np.random.choice(available_E, size=N_EE, replace=False)
            pre_EE_list.append(j_pre_EE)
            post_EE_list.append(np.full(N_EE, j_post))
            
            # For inhibitory-to-excitatory connections:
            j_pre_EI = np.random.choice(np.arange(N_I), size=N_EI, replace=False)
            pre_EI_list.append(j_pre_EI)
            post_EI_list.append(np.full(N_EI, j_post))
            
        pre_EE = np.concatenate(pre_EE_list)
        post_EE = np.concatenate(post_EE_list)
        pre_EI = np.concatenate(pre_EI_list)
        post_EI = np.concatenate(post_EI_list)
        
        # Build sparse connectivity matrices:
        C_EE = sparse.coo_matrix((np.ones_like(pre_EE), (post_EE, pre_EE)), shape=(N_E, N_E))
        # For C_EI, note that MATLAB’s comment indicates this is "I-to-E connectivity"
        # (i.e. postsynaptic excitatory neurons receiving connections from inhibitory neurons).
        C_EI = sparse.coo_matrix((np.ones_like(pre_EI), (post_EI, pre_EI)), shape=(N_E, N_I))

        # For postsynaptic inhibitory neurons:
        N_II = int(round(N_I * p_II))      # number of inhibitory presynaptic connections to an inhibitory neuron
        N_IE = int(round(N_E * p_IE))        # number of excitatory presynaptic connections to an inhibitory neuron
        
        pre_II_list, post_II_list = [], []
        pre_IE_list, post_IE_list = [], []
        for j_post in range(N_I):
            available_I = np.delete(np.arange(N_I), j_post)
            j_pre_II = np.random.choice(available_I, size=N_II, replace=False)
            pre_II_list.append(j_pre_II)
            post_II_list.append(np.full(N_II, j_post))
            
            j_pre_IE = np.random.choice(np.arange(N_E), size=N_IE, replace=False)
            pre_IE_list.append(j_pre_IE)
            post_IE_list.append(np.full(N_IE, j_post))
            
        pre_II = np.concatenate(pre_II_list)
        post_II = np.concatenate(post_II_list)
        pre_IE = np.concatenate(pre_IE_list)
        post_IE = np.concatenate(post_IE_list)
        
        C_II = sparse.coo_matrix((np.ones_like(pre_II), (post_II, pre_II)), shape=(N_I, N_I))
        # Here C_IE represents "E-to-I connectivity" (postsynaptic inhibitory neurons receiving from excitatory)
        C_IE = sparse.coo_matrix((np.ones_like(pre_IE), (post_IE, pre_IE)), shape=(N_I, N_E))
    
    # Define E-to-E forward inhibition connectivity as in MATLAB:
    # C_EE_FFI = C_EI * C_IE
    # (Note: the product of two sparse matrices returns a sparse matrix.)
    if connect_type == 1:
        C_EE_FFI = C_EI @ C_IE
    else:
        C_EE_FFI = C_EI @ C_IE

    # -------------------------
    # PARAMETERS
    # -------------------------
    coba = 0              # 0 for CUBA; 1 for COBA (not implemented)
    R_m = 1e5             # membrane resistance (Ohm)
    C_m = 2e-8            # membrane capacitance (F)
    tau_e = 0.002         # excitatory synaptic time constant (s)
    tau_i = 0.002         # inhibitory synaptic time constant (s)
    V_rest = -0.06        # resting membrane potential (V)
    V_reset = -0.06       # reset potential (V)
    V_thr = -0.055        # firing threshold (V)
    V_peak = -0.03        # peak membrane potential (V) (used only for display)
    THETA = 53e-9         # constant injected current (A)
    refract_period = 0.005  # absolute refractory period (s)
    wEEmax = 1            # maximum value for E-to-E weight
    
    # CUBA parameters:
    EPSC_peak = 0.6e-9    # peak EPSC amplitude (A)
    IPSC_peak = -5.0e-9   # peak IPSC amplitude (A)

    tau_m = R_m * C_m     # membrane time constant (s)
    
    # "Display neurons" (indices for detailed display, adjusted to 0-indexing)
    disp_neurons = [0, 10, 110, 120]
    
    # Refractory period vector (one value per neuron)
    refract_p_rand = np.ones(N) * refract_period
    
    total_time = 0  # initial simulated time

    # -------------------------
    # SET UP SEED GROUPS
    # -------------------------
    n_seed = 0       # number of seed groups (set to 0 if not used)
    s_seed = 10      # size of each seed group
    seed_group = []
    for sg in range(n_seed):
        seed_group.append(np.arange(sg * s_seed, sg * s_seed + s_seed))
    ssn = np.arange(n_seed * s_seed)
    
    # Remove all seed-to-seed connections in C_EE.
    if ssn.size > 0 and connect_type == 2:
        C_EE = C_EE.tolil()
        C_EE[np.ix_(ssn, ssn)] = 0
        C_EE = C_EE.tocoo()
    
    # -------------------------
    # INITIALIZATION
    # -------------------------
    # Initial membrane potentials: V = V_rest + noise.
    V = V_rest * (1 + 0.1 * np.random.randn(N))
    refractory_counter = np.zeros(N)
    
    spiking_e = np.zeros(N_E, dtype=bool)  # spiking E neurons
    spiking_i = np.zeros(N_I, dtype=bool)  # spiking I neurons
    
    curr_ee = np.zeros(N_E)  # E-to-E currents (excitatory population)
    curr_ie = np.zeros(N_I)  # E-to-I currents (inhibitory population)
    curr_ai = np.zeros(N)    # I-to-all currents (total for all neurons)
    
    # STDP memory variables (for excitatory neurons)
    P = np.zeros(N_E)
    D = np.zeros(N_E)
    
    spike_counts = np.zeros(N)
    
    # -------------------------
    # WEIGHT INITIALIZATION
    # -------------------------
    # Initialize E-to-E weights based on connectivity.
    C_EE_dense = C_EE.toarray() if hasattr(C_EE, "toarray") else C_EE
    w_EE = (0.45 + 0.1 * np.random.rand(N_E, N_E)) * C_EE_dense
    
    # For the other weights, we keep the connectivity structure.
    # For E-to-I, convert sparse matrix to dense if necessary.
    if connect_type == 2:
        w_IE = C_IE.toarray() if hasattr(C_IE, "toarray") else C_IE
    else:
        w_IE = None  # or however you want to handle the ER case
    
    # I-to-all weights: vertical stack of C_EI (I→E) and C_II (I→I)
    if connect_type == 2:
        w_AI = sparse.vstack([C_EI, C_II])
    else:
        w_AI = None

    # -------------------------
    # MOTIF ANALYSIS INITIALIZATION
    # -------------------------
    FFI_thr_values = [15, 17, 20, 23, 25, 27, 30]
    bpm_count = []
    bpm_rand_count = []
    bfm_count = []
    bfm_rand_count = []

    # Ensure w_EE is in dense array format.
    w_EE = np.array(w_EE)
    
    # -------------------------
    # END SETUP AND PACK VARIABLES
    # -------------------------
    
    setup_dict = {
        "N": N,
        "N_E": N_E,
        "N_I": N_I,
        "ee": ee,
        "ii": ii,
        "C_EE": C_EE,
        "C_IE": C_IE if connect_type == 2 else None,
        "C_EI": C_EI if connect_type == 2 else None,
        "C_II": C_II if connect_type == 2 else None,
        "C_EE_FFI": C_EE_FFI,
        "R_m": R_m,
        "C_m": C_m,
        "tau_e": tau_e,
        "tau_i": tau_i,
        "V_rest": V_rest,
        "V_reset": V_reset,
        "V_thr": V_thr,
        "V_peak": V_peak,
        "THETA": THETA,
        "refract_period": refract_period,
        "wEEmax": wEEmax,
        "EPSC_peak": EPSC_peak,
        "IPSC_peak": IPSC_peak,
        "tau_m": tau_m,
        "disp_neurons": disp_neurons,
        "refract_p_rand": refract_p_rand,
        "total_time": total_time,
        "seed_group": seed_group,
        "ssn": ssn,
        "V": V,
        "refractory_counter": refractory_counter,
        "spiking_e": spiking_e,
        "spiking_i": spiking_i,
        "curr_ee": curr_ee,
        "curr_ie": curr_ie,
        "curr_ai": curr_ai,
        "P": P,
        "D": D,
        "spike_counts": spike_counts,
        "w_EE": w_EE,
        "w_IE": w_IE,
        "w_AI": w_AI,
        "FFI_thr_values": FFI_thr_values,
        "bpm_count": bpm_count,
        "bpm_rand_count": bpm_rand_count,
        "bfm_count": bfm_count,
        "bfm_rand_count": bfm_rand_count,
        "CCG_flag_pop": 1,
        "CCG_flag_pair": 1
    }
    
    return setup_dict

def run_stdp(setup, run_length, dt=0.0005, L_analysis=5.0, w_distr_prctile=99, f_seed_act=0):
    """
    Continues an STDP simulation by modifying a deep copy of the provided setup dictionary.
    
    If the provided dictionary already contains simulation data (indicated by the key "T_sim"),
    the new simulation run (of duration run_length) is appended to the existing data.
    
    Parameters:
      setup         : dict
                      Dictionary containing network parameters and initial state (from setup_stdp).
      run_length    : float
                      Duration (in seconds) of the additional simulation run (default: 10.0 s).
      dt            : float
                      Simulation time step in seconds (default: 0.0005 s).
      L_analysis    : float
                      Duration (in seconds) for detailed display data recording (default: 5.0 s).
      w_distr_prctile : float
                      Percentile threshold for weight binarization in motif analysis (default: 99).
      f_seed_act    : float
                      Frequency (Hz) for seed group activation (default: 0).
    
    Returns:
      state         : dict
                      A deep-copied and updated version of the setup dictionary that now contains the additional
                      simulation run data. If simulation data already existed, the new data are appended.
    """
    # Create a deep copy of the input dictionary.
    state = copy.deepcopy(setup)
    
    # Ensure motif-analysis keys are lists (so that we can append new results).
    if "bfm_count" not in state or not isinstance(state["bfm_count"], list):
        state["bfm_count"] = []
        state["bfm_rand_count"] = []
        state["bpm_count"] = []
        state["bpm_rand_count"] = []
    
    # Store simulation parameters into the state.
    state["dt"] = dt
    state["L_analysis"] = L_analysis
    state["run_length"] = run_length
    state["w_distr_prctile"] = w_distr_prctile

    # Determine how many time steps have already been simulated.
    if "T_sim" in state:
        T_prev = int(state["T_sim"])
    else:
        T_prev = 0

    # Calculate the number of new time steps.
    T_new = int(round(run_length / dt))
    T_total = T_prev + T_new
    state["T_sim"] = T_total

    # Unpack state variables.
    curr_ee = state["curr_ee"]
    curr_ie = state["curr_ie"]
    curr_ai = state["curr_ai"]
    V = state["V"]
    refractory_counter = state["refractory_counter"]
    spiking_e = state["spiking_e"]
    spiking_i = state["spiking_i"]
    w_EE = state["w_EE"]
    w_IE = state["w_IE"]
    w_AI = state["w_AI"]
    C_EE = state["C_EE"]
    C_EE_FFI = state["C_EE_FFI"]
    P = state["P"]
    D = state["D"]
    ee = state["ee"]
    ii = state["ii"]
    N_E = state["N_E"]
    N_I = state["N_I"]
    disp_neurons = state["disp_neurons"]
    seed_group = state["seed_group"]
    ssn = state["ssn"]
    tau_e = state["tau_e"]
    tau_i = state["tau_i"]
    EPSC_peak = state["EPSC_peak"]
    IPSC_peak = state["IPSC_peak"]
    R_m = state["R_m"]
    THETA = state["THETA"]
    tau_m = state["tau_m"]
    V_rest = state["V_rest"]
    V_reset = state["V_reset"]
    V_thr = state["V_thr"]
    V_peak = state["V_peak"]
    wEEmax = state["wEEmax"]
    refract_p_rand = state["refract_p_rand"]
    total_time = state["total_time"]
    FFI_thr_values = state["FFI_thr_values"]

    # STDP parameters: use defaults if not present.
    tau_P = state.get("tau_P", 0.002)
    tau_D = state.get("tau_D", 0.005)
    A_P = state.get("A_P", 0.001)
    A_D = state.get("A_D", -1.2 * 0.001 * 0.002 / 0.005)
    
    # Derived simulation parameters.
    L_analysis_dt = int(round(L_analysis / dt))
    refract_dt = np.round(refract_p_rand / dt).astype(int)
    p_seed_act = dt * f_seed_act

    N = V.shape[0]
    
    # Retrieve old simulation-recording data, if any.
    old_spikes = state.get("spikes", None)
    old_f_rate_E = state.get("f_rate_E", None)
    old_f_rate_I = state.get("f_rate_I", None)
    old_spike_counts = state.get("spike_counts", None)
    old_disp_V = state.get("disp_V", None)
    old_disp_curr_e = state.get("disp_curr_e", None)
    old_disp_curr_i = state.get("disp_curr_i", None)
    
    # Preallocate new arrays for the new simulation run.
    new_spikes = np.zeros((N, T_new), dtype=bool)
    new_f_rate_E = np.zeros(T_new)
    new_f_rate_I = np.zeros(T_new)
    new_spike_counts = np.zeros(N)
    new_disp_V = np.zeros((len(disp_neurons), L_analysis_dt))
    new_disp_curr_e = np.zeros((len(disp_neurons), L_analysis_dt))
    new_disp_curr_i = np.zeros((len(disp_neurons), L_analysis_dt))
    t_last_act = 0

    # Main simulation loop for new time steps.
    for t in range(T_new):
        # --- Update synaptic currents.
        curr_ee = (1 - dt / tau_e) * curr_ee + EPSC_peak * (w_EE @ spiking_e.astype(float))
        curr_ie = (1 - dt / tau_e) * curr_ie + EPSC_peak * (w_IE @ spiking_e.astype(float))
        curr_ae = np.concatenate([curr_ee, curr_ie])
        curr_ai = (1 - dt / tau_i) * curr_ai + IPSC_peak * (w_AI @ spiking_i.astype(float))
    
        # --- Update membrane potential.
        dV_dt = (V_rest - V + R_m * (THETA + curr_ae + curr_ai)) / tau_m
        V = V + dV_dt * dt
    
        # --- Spike detection and refractory handling.
        refractory = refractory_counter > 0
        spiking = (V >= V_thr) & (~refractory)
    
        # Activate a seed group if the conditions are met.
        if (t > t_last_act + int(0.1 / dt)) and (np.random.rand() < p_seed_act):
            t_last_act = t
            if len(seed_group) > 0:
                sg = np.random.randint(len(seed_group))
                act_neur = seed_group[sg]
                spiking[act_neur] = True  # force activation
    
        V[spiking] = V_peak
        refractory_counter = refractory_counter - 1
        V[refractory] = V_reset
        refractory_counter[spiking] = refract_dt[spiking]
    
        # Update spiking sub-vectors.
        spiking_e = spiking[ee]
        spiking_i = spiking[ii]
    
        new_f_rate_E[t] = np.sum(spiking_e) / N_E
        new_f_rate_I[t] = np.sum(spiking_i) / N_I
        new_spike_counts += spiking.astype(int)
    
        # Record display data (only for the first L_analysis_dt time steps).
        if t < L_analysis_dt:
            new_spikes[:, t] = spiking
            new_disp_V[:, t] = V[disp_neurons]
            new_disp_curr_e[:, t] = curr_ae[disp_neurons]
            new_disp_curr_i[:, t] = curr_ai[disp_neurons]
    
        # --- Motif Analysis (every 10 seconds of cumulative simulation) ---
        if abs(((T_prev + t) * dt) % 10) < 1e-10:
            if hasattr(C_EE, "toarray"):
                mask_C_EE = (C_EE.toarray() != 0)
            else:
                mask_C_EE = (C_EE != 0)
            WEE_current = w_EE[mask_C_EE]
            thr_W = np.percentile(WEE_current, w_distr_prctile)
            B = w_EE >= thr_W
            B_rand = randomize_graph_2(B, int(1e4))
    
            # Append new motif analysis results.
            state["bfm_count"].append(motiv4(B))
            state["bfm_rand_count"].append(motiv4(B_rand))
    
            temp_bpm = []
            temp_bpm_rand = []
            for FFI_thr in FFI_thr_values:
                temp_bpm.append(motiv3(B, C_EE_FFI, FFI_thr))
                temp_bpm_rand.append(motiv3(B_rand, C_EE_FFI, FFI_thr))
            state["bpm_count"].append(temp_bpm)
            state["bpm_rand_count"].append(temp_bpm_rand)
    
        # --- STDP Updates.
        P = P * (1 - dt / tau_P) + spiking_e.astype(float)
        D = D * (1 - dt / tau_D) + spiking_e.astype(float)
        idx_spiking_e = np.where(spiking_e)[0]
        if idx_spiking_e.size > 0:
            w_EE[idx_spiking_e, :] += A_P * P[np.newaxis, :]
            w_EE[:, idx_spiking_e] += A_D * D[:, np.newaxis]
        np.fill_diagonal(w_EE, 0)
        if ssn.size > 0:
            w_EE[np.ix_(ssn, ssn)] = 0
        if abs(((T_prev + t) * dt) % 1) < 1e-10:
            w_EE = np.clip(w_EE, 0, wEEmax) # Clip weights
        if abs(((T_prev + t) * dt) % 10) < 1e-10:
            print(f"{(T_prev + t)*dt:.1f} sec") # Print biological time
    
    total_time += run_length

    # Concatenate new simulation-recording arrays with previous ones, if they exist.
    if old_spikes is not None:
        state["spikes"] = np.hstack([old_spikes, new_spikes])
    else:
        state["spikes"] = new_spikes

    if old_f_rate_E is not None:
        state["f_rate_E"] = np.concatenate([old_f_rate_E, new_f_rate_E])
    else:
        state["f_rate_E"] = new_f_rate_E

    if old_f_rate_I is not None:
        state["f_rate_I"] = np.concatenate([old_f_rate_I, new_f_rate_I])
    else:
        state["f_rate_I"] = new_f_rate_I

    if old_spike_counts is not None:
        state["spike_counts"] = old_spike_counts + new_spike_counts
    else:
        state["spike_counts"] = new_spike_counts

    if old_disp_V is not None:
        state["disp_V"] = np.hstack([old_disp_V, new_disp_V])
    else:
        state["disp_V"] = new_disp_V

    if old_disp_curr_e is not None:
        state["disp_curr_e"] = np.hstack([old_disp_curr_e, new_disp_curr_e])
    else:
        state["disp_curr_e"] = new_disp_curr_e

    if old_disp_curr_i is not None:
        state["disp_curr_i"] = np.hstack([old_disp_curr_i, new_disp_curr_i])
    else:
        state["disp_curr_i"] = new_disp_curr_i

    # Update remaining state variables.
    state["curr_ee"] = curr_ee
    state["curr_ie"] = curr_ie
    state["curr_ai"] = curr_ai
    state["V"] = V
    state["refractory_counter"] = refractory_counter
    state["w_EE"] = w_EE
    state["P"] = P
    state["D"] = D
    state["total_time"] = total_time

    return state

def spike_train_xcorr(s_times_1, s_train_2, lags):
    """
    Computes cross-correlation histogram for two point processes.
    
    Parameters:
      s_times_1 : 1D array-like of spike times (indices) for neuron 1.
      s_train_2 : 1D numpy array of 0/1 values (length T) for neuron 2.
      lags      : 1D array of integer lags.
      
    Returns:
      crossco  : 1D numpy array of the conditional firing rate of neuron 2 at each lag,
                 computed as (# spikes at that lag) / (# spikes of neuron 1 [with appropriate lag]).
    """
    T = len(s_train_2)
    crossco = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        lst = np.array(s_times_1) + lag
        lst = lst[(lst >= 0) & (lst < T)]
        if len(lst) > 0:
            crossco[i] = np.sum(s_train_2[lst]) / (len(lst) + np.finfo(float).eps)
        else:
            crossco[i] = 0
    return crossco

def analyze_stdp(state, save_fig_dir: str='Figures'):
    """
    Performs analysis for an STDP simulation. In addition to the initial analyses (raster, 
    firing rates, smoothing, z-transformation, histograms, ISI/CV and membrane potential plots),
    this function now also:
      (a) Plots a histogram of E-to-E weights,
      (b) Plots a 2D histogram of the in- vs. out-degrees (after binarizing weights),
      (c) Shows a binarized graph (via show_binarized_graph) and a spy plot,
      (d) Plots motif counts (BPM and BFM) versus time for different FFI thresholds,
      (e) Displays histograms of length-2, 3, and 4 parallel motifs (using matrix powers),
      (f) Optionally computes population cross-correlograms (CCGs) for total E/I activity,
      (g) Optionally computes average CCGs for pairs of excitatory neurons (connected vs. unconnected).
    
    It assumes that state is a dictionary that includes simulation results and parameters such as:
      "dt", "L_analysis", "w_EE", "C_EE", "wEEmax", "total_time", "w_distr_prctile",
      "bpm_count", "bpm_rand_count", "bfm_count", "bfm_rand_count", "FFI_thr_values",
      "CCG_flag_pop", and "CCG_flag_pair".
    
    All plots are generated using matplotlib.
    """

    # --- Use dt from state ---
    dt = state["dt"]

    # Create figure save directory
    os.makedirs(save_fig_dir, exist_ok=True)
    fig_path = lambda x: os.path.join(save_fig_dir, x)
    
    # (Assuming the earlier analysis has been run and variables computed:)
    # For example, assume the following variables were computed in the first part:
    #   f_rate_E, f_rate_I, spikes, spike_counts, ifrE2, ifrI2, t_plot, disp_neurons, etc.
    # Here we retrieve what we need from state.
    w_EE = state["w_EE"]
    C_EE = state["C_EE"]
    wEEmax = state["wEEmax"]
    total_time = state["total_time"]
    w_distr_prctile = state["w_distr_prctile"]
    FFI_thr_values = state["FFI_thr_values"]
    bpm_count = state["bpm_count"]
    bpm_rand_count = state["bpm_rand_count"]
    bfm_count = state["bfm_count"]
    bfm_rand_count = state["bfm_rand_count"]

    # For demonstration, assume that prior analysis computed "ifrE2" and "ifrI2"
    # (e.g., 3ms-smoothed firing rates) and stored them in state.
    if "ifrE2" in state:
        ifrE2 = state["ifrE2"]
    else:
        # Otherwise, use f_rate_E itself as a placeholder.
        ifrE2 = state["f_rate_E"]
    if "ifrI2" in state:
        ifrI2 = state["ifrI2"]
    else:
        ifrI2 = state["f_rate_I"]

    # ---------------------------
    # (a) Histogram of E-to-E weights.
    # ---------------------------
    plt.figure()
    # Ensure C_EE is dense for logical indexing.
    if hasattr(C_EE, "toarray"):
        C_EE_dense = C_EE.toarray()
    else:
        C_EE_dense = C_EE
    WEE = w_EE[np.asarray(C_EE_dense, dtype=bool)]
    bins = np.linspace(0, wEEmax, 100)
    plt.hist(WEE.ravel(), bins=bins, color='skyblue', edgecolor='black')
    plt.gca().tick_params(labelsize=16)
    plt.grid(True)
    plt.xlim([0, wEEmax])
    plt.xlabel('excitatory weight', fontsize=16)
    plt.ylabel('# synapses', fontsize=16)
    plt.title(f'{total_time:.2f} sec', fontsize=16)
    plt.savefig(fig_path('e_to_e_weights.pdf'))
    plt.close()
    
    # ---------------------------
    # (b) 2D histogram of IN vs. OUT binarized weights.
    # ---------------------------
    thr_wEE = np.percentile(WEE, w_distr_prctile)
    w_EE_discrete = w_EE >= thr_wEE
    # number_outgoing: sum along axis 0; number_incoming: sum along axis 1.
    number_outgoing = np.sum(w_EE_discrete, axis=0)
    number_incoming = np.sum(w_EE_discrete, axis=1)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    # Use matplotlib’s hist2d.
    plt.hist2d(number_incoming, number_outgoing, bins=50, cmap='viridis')
    mean_in = np.mean(number_incoming)
    mean_out = np.mean(number_outgoing)
    plt.xlabel(f'# IN edges -- Mean = {mean_in:.2f}', fontsize=16)
    plt.ylabel(f'# OUT edges -- Mean = {mean_out:.2f}', fontsize=16)
    plt.gca().tick_params(labelsize=16)
    plt.subplot(1, 2, 2)
    plt.scatter(number_outgoing, number_incoming, c='purple', edgecolors='k')
    plt.gca().tick_params(labelsize=16)
    plt.grid(True)
    plt.xlabel('# OUT edges', fontsize=16)
    plt.ylabel('# IN edges', fontsize=16)
    plt.title(f'{total_time:.2f} sec', fontsize=16)
    plt.savefig(fig_path('in_vs_out_binarized_weights.pdf'))
    plt.close()
    
    # ---------------------------
    # (c) Show binarized graph and a spy plot.
    # ---------------------------
    fig = binarized_graph(w_EE, thr_wEE)
    fig.savefig(fig_path('binarized_graph.pdf'))
    plt.close(fig)
    
    plt.figure()
    plt.spy(w_EE >= thr_wEE)
    plt.gca().tick_params(labelsize=16)
    plt.title('Spy plot of binarized E-to-E weights', fontsize=16)
    plt.savefig(fig_path('spy_e_to_e.pdf'))
    plt.close()
    
    # ---------------------------
    # (d) Plot motif count evolution.
    # ---------------------------
    # Assume bpm_count and bpm_rand_count are arrays or lists.
    lbfmc = len(bfm_count)  # number of motif analyses over time
    FFI_thr_vals = FFI_thr_values  # for looping
    Ftv = 0
    subp = 0
    bpm_count = np.array(bpm_count)
    bpm_rand_count = np.array(bpm_rand_count)
    n_FFI = bpm_count.shape[0]
    
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(fig_path('motif_count_evolution.pdf')) as pdf:
        fig = plt.figure(figsize=(10, 4))
        # Loop over each FFI threshold (for BPM counts).
        for Ftv in range(n_FFI):
            subp += 1
            plt.subplot(2, 2, subp)
            x_axis = 10 * np.arange(1, bpm_count.shape[1] + 1)
            plt.plot(x_axis, bpm_count[Ftv, :], 'r-o', linewidth=2, label='STDP graph')
            plt.plot(x_axis, bpm_rand_count[Ftv, :], 'b-o', linewidth=2, label='randomized')
            plt.gca().tick_params(labelsize=16)
            plt.xlabel('time', fontsize=16)
            plt.ylabel('BPM count', fontsize=16)
            plt.title(f'FFI threshold: {FFI_thr_vals[Ftv]}', fontsize=16)
            plt.legend(loc='best')
            plt.grid(True)
            
            # Every 4 subplots (and if there are more thresholds to plot), save the current figure and start a new one.
            if (Ftv + 1) % 4 == 0 and (Ftv + 1) < n_FFI:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(10, 4))
                subp = 0  # reset the subplot count for the new figure
                
        # After looping over the FFI thresholds, add one extra subplot for the BFM counts.
        subp += 1
        plt.subplot(2, 2, subp)
        # Here we assume that bfm_count and bfm_rand_count are 1D arrays or lists (one value per motif analysis)
        x_axis = 10 * np.arange(1, len(bfm_count) + 1)
        plt.plot(x_axis, bfm_count, 'r-o', linewidth=2, label='STDP graph')
        plt.plot(x_axis, bfm_rand_count, 'b-o', linewidth=2, label='randomized')
        plt.gca().tick_params(labelsize=16)
        plt.xlabel('time', fontsize=16)
        plt.ylabel('BFM count', fontsize=16)
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save the final figure.
        pdf.savefig(fig)
        plt.close(fig)
    
    # ---------------------------
    # (e) 2D histograms of length-2, 3, and 4 parallel motifs.
    # ---------------------------
    # Compute B from w_EE using the binarization threshold.
    B = (w_EE >= thr_wEE).astype(int)
    B_rand = randomize_graph_2(B, int(1e4)).toarray()
    # Compute matrix powers.
    b2 = np.linalg.matrix_power(B, 2)
    b2r = np.linalg.matrix_power(B_rand, 2)
    b2p = (b2 > 0) & (b2r > 0)
    b2_vals = b2[b2p]
    b2r_vals = b2r[b2p]
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    bins_b2 = np.arange(0, 51, 1)
    axs[0, 0].hist([b2_vals, b2r_vals], bins=bins_b2, label=['STDP Graph', 'Randomized'])
    axs[0, 0].tick_params(labelsize=18)
    axs[0, 0].set_xlim([20, 50])
    axs[0, 0].set_xlabel("k", fontsize=18)
    axs[0, 0].set_ylabel("count", fontsize=18)
    axs[0, 0].set_title("LENGTH-2 k-PARALLEL MOTIFS", fontsize=18)
    
    b3 = np.linalg.matrix_power(B, 3)
    b3r = np.linalg.matrix_power(B_rand, 3)
    b3p = (b3 > 0) & (b3r > 0)
    b3_vals = b3[b3p]
    b3r_vals = b3r[b3p]
    bins_b3 = np.arange(0, 2001, 10)
    axs[0, 1].hist([b3_vals, b3r_vals], bins=bins_b3, label=['STDP Graph', 'Randomized'])
    axs[0, 1].tick_params(labelsize=18)
    axs[0, 1].set_xlim([800, 2000])
    axs[0, 1].set_xlabel("k", fontsize=18)
    axs[0, 1].set_ylabel("count", fontsize=18)
    axs[0, 1].set_title("LENGTH-3 k-PARALLEL MOTIFS", fontsize=18)
    
    b4 = np.linalg.matrix_power(B, 4)
    b4r = np.linalg.matrix_power(B_rand, 4)
    b4p = (b4 > 0) & (b4r > 0)
    b4_vals = b4[b4p]
    b4r_vals = b4r[b4p]
    bins_b4 = np.arange(0, 60001, 500)
    axs[1, 0].hist([b4_vals, b4r_vals], bins=bins_b4, label=['STDP Graph', 'Randomized'])
    axs[1, 0].tick_params(labelsize=18)
    axs[1, 0].set_xlim([40000, 60000])
    axs[1, 0].set_xlabel("k", fontsize=18)
    axs[1, 0].set_ylabel("count", fontsize=18)
    axs[1, 0].set_title("LENGTH-4 k-PARALLEL MOTIFS", fontsize=18)

    axs[1, 1].axis('off')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[1, 1].legend(handles, labels, loc='center', fontsize=18)
    fig.savefig(fig_path('k-Parallel Motifs.pdf'))
    plt.close(fig)
    
    # ---------------------------
    # (f) Compute population CCGs and ACGs (if enabled)
    # ---------------------------
    CCG_flag_pop = state.get('CCG_flag_pop', 0)
    if CCG_flag_pop == 1:
        # Create a 2×2 subplot layout
        fig, axs = plt.subplots(2, 2)

        # Subplot 1: short lag (50 bins) for CCG: Total E to Total I
        lags = np.arange(-50, 51)
        r = np.correlate(ifrE2 - np.mean(ifrE2), ifrI2 - np.mean(ifrI2), mode='full')
        center = len(r) // 2
        r_short = r[center-50:center+51]
        axs[0, 0].plot(-lags * dt, r_short, linewidth=2)
        axs[0, 0].tick_params(labelsize=16)
        axs[0, 0].set_title('CCG: Total E to Total I', fontsize=16)
        axs[0, 0].set_xlabel('lag (s)', fontsize=16)
        axs[0, 0].grid(True)

        # Subplot 2: long lag (1000 bins) for CCG: Total E to Total I
        lags_long = np.arange(-1000, 1001)
        r = np.correlate(ifrE2 - np.mean(ifrE2), ifrI2 - np.mean(ifrI2), mode='full')
        center = len(r) // 2
        r_long = r[center-1000:center+1001]
        axs[0, 1].plot(-lags_long * dt, r_long, linewidth=2)
        axs[0, 1].tick_params(labelsize=16)
        axs[0, 1].set_title('CCG: Total E to Total I', fontsize=16)
        axs[0, 1].set_xlabel('lag (s)', fontsize=16)
        axs[0, 1].grid(True)

        # Subplot 3: ACG: Total E
        r = np.correlate(ifrE2 - np.mean(ifrE2), ifrE2 - np.mean(ifrE2), mode='full')
        center = len(r) // 2
        r_acg_E = r[center-1000:center+1001]
        axs[1, 0].plot(-lags_long * dt, r_acg_E, linewidth=2)
        axs[1, 0].tick_params(labelsize=16)
        axs[1, 0].set_title('ACG: Total E', fontsize=16)
        axs[1, 0].set_xlabel('lag (s)', fontsize=16)
        axs[1, 0].grid(True)

        # Subplot 4: ACG: Total I
        r = np.correlate(ifrI2 - np.mean(ifrI2), ifrI2 - np.mean(ifrI2), mode='full')
        center = len(r) // 2
        r_acg_I = r[center-1000:center+1001]
        axs[1, 1].plot(-lags_long * dt, r_acg_I, linewidth=2)
        axs[1, 1].tick_params(labelsize=16)
        axs[1, 1].set_title('ACG: Total I', fontsize=16)
        axs[1, 1].set_xlabel('lag (s)', fontsize=16)
        axs[1, 1].grid(True)

        # Tight layout, then save the figure as PDF and close it.
        fig.tight_layout()
        fig.savefig(fig_path('population_ccg_acg.pdf'))
        plt.close(fig)
    
    # ---------------------------
    # (g) Compute average CCGs between excitatory neuron pairs (if enabled)
    # ---------------------------
    CCG_flag_pair = state.get("CCG_flag_pair", 0)
    if CCG_flag_pair == 1:
        lags = np.arange(-200, 201)
        n_pairs = int(1e3)
        N_E = state["N_E"]
        # We'll assume spike_times is stored in state from earlier analysis.
        spike_times = state.get("spike_times", None)
        if spike_times is None:
            # If not, compute spike_times from spikes.
            s_stamps, t_stamps = np.nonzero(state["spikes"])
            spike_times = [t_stamps[s_stamps == n] for n in range(state["spikes"].shape[0])]
        n_p_unconnected = 0
        ccg_unconnected = np.zeros(len(lags))
        C_EE_dense = C_EE.toarray()  if hasattr(C_EE, "toarray") else C_EE
        while n_p_unconnected < n_pairs:
            a = np.random.randint(0, N_E)
            b = np.random.randint(0, N_E)
            if (a != b) and (C_EE_dense[b, a] == 0) and (C_EE_dense[a, b] == 0):
                n_p_unconnected += 1
                train_b = np.zeros(state["spikes"].shape[1])
                # Use spike_times[b] as indices.
                if len(spike_times[b]) > 0:
                    train_b[spike_times[b]] = 1
                ccg = spike_train_xcorr(spike_times[a], train_b, lags)
                ccg_unconnected += ccg
        n_p_connected = 0
        ccg_connected = np.zeros(len(lags))
        while n_p_connected < n_pairs:
            a = np.random.randint(0, N_E)
            b = np.random.randint(0, N_E)
            if (a != b) and (C_EE_dense[b, a] == 0) and (C_EE_dense[a, b] > 0):
                n_p_connected += 1
                train_b = np.zeros(state["spikes"].shape[1])
                if len(spike_times[b]) > 0:
                    train_b[spike_times[b]] = 1
                ccg = spike_train_xcorr(spike_times[a], train_b, lags)
                ccg_connected += ccg

        # Create a figure with two subplots (vertically aligned).
        fig, axs = plt.subplots(2, 1)

        # First subplot: average CCG for connected pairs.
        axs[0].plot(-lags * dt * 1e3, ccg_connected / (n_p_connected * dt), linewidth=2)
        axs[0].tick_params(labelsize=16)
        axs[0].grid(True)
        axs[0].set_ylabel('conditional firing rate (Hz)', fontsize=16)
        axs[0].legend([f'{n_p_connected} connected pairs'])
        axs[0].set_title('aver. CCG for connected pairs', fontsize=16)

        # Second subplot: average CCG for unconnected pairs.
        axs[1].plot(-lags * dt * 1e3, ccg_unconnected / (n_p_unconnected * dt), linewidth=2)
        axs[1].tick_params(labelsize=16)
        axs[1].grid(True)
        axs[1].set_xlabel('time (ms)', fontsize=16)
        axs[1].set_ylabel('conditional firing rate (Hz)', fontsize=16)
        axs[1].legend([f'{n_p_unconnected} unconnected pairs'])
        axs[1].set_title('aver. CCG for unconnected pairs', fontsize=16)

        # Adjust layout and save the figure as a PDF.
        fig.tight_layout()
        fig.savefig(fig_path('average_ccg.pdf'))
        plt.close(fig)

def main(N: int, run_length: int):
    setup = setup_stdp(N=N)
    sim_state = run_stdp(setup=setup, run_length=run_length)
    os.makedirs('States', exist_ok=True)
    with open(os.path.join('States', f'sim_state_{N}_{run_length}.pkl'), 'wb') as file:
        pickle.dump(sim_state, file)
    analyze_stdp(sim_state)

def parse_args():
    parser = argparse.ArgumentParser(description='Run STDP simulation')
    parser.add_argument('-n', '--neurons', type=int, required=True,
                        help='Number of excitatory neurons')
    parser.add_argument('-t', '--run_length', type=int, required=True,
                        help='Run length in seconds')
    args = parser.parse_args()
    return args.neurons, args.run_length

if __name__ == '__main__':
    main(*parse_args())
