# --------------------------------------------------------------------------------------
# Generating signal waveforms with predefined correlations
#
# For detailed description of the method see:
#
#   A. Moiseev, "Simulating brain signals with predened mutual correlations"
#   doi:
#
# A.Moiseev, Behavioral and Cognitive Neuroscience Institute,
# Simon Fraser University, Canada
# May 2021
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Example
# --------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from gen_corr_src_set import gen_band_noise, gen_multi_band_noise, gen_corr_src_set

fs = 200.
n = 6       # Number of signals
m = 200     # Number of time points: 1 second per epoch

# Frequency bands specification
bands = ((8.,12.), (8.,12.),        # A pair of alpha band signals
        (16.,20.), (16.,20.),       # A pair of beta-band signals
        (35.,40.), (35.,40.))       # A pair of gamma-band signals

# taper: hanning * exp(-6t/T)
taper = np.hanning(m) * np.exp(-6. * np.array(range(m))/m)

# Evoked signals specification
t = np.linspace(0, m/fs, m, endpoint=False)
ev_alpha = np.sin(6.*np.pi*t*fs/m) * taper  # alpha signals evoked parts
ev_beta = -np.hanning(m) * taper           # beta signals evoked parts

evoked = np.zeros((n,m))
evoked[0,:] = ev_alpha;         # Evoked alpha's are anti-correlated
evoked[1,:] = -ev_alpha;
evoked[2,:] = ev_beta;          # Evoked beta are correlated 
evoked[3,:] = ev_beta;          # Gamma's have no evoked parts

# Evoked parts contributions to the whole signals
quot = np.ones(n) * 0.6

corr_alpha = -0.9   # Correlation within the alpha pair
corr_beta = 0.5;    # Correlation within the beta pair
corr_gamma = 0.2;   # Correlation within the gamma pair
corr_ab = 0.;       # Correlation between alpha and beta pairs
corr_ag = 0.33;     # Correlation between alpha and gamma pairs
corr_bg = 0.;       # Correlation between beta and gamma pairs


order = 4                           # Band pass filter half-order
rand_seed = 210530                  # Seed the random generator for repeatable results
pngname = 'signals.png'
dpi_res = 300                       # Set DPI resolution for the saved figure, if necessary

target_corr = np.identity(n)
# Within pairs correlations
target_corr[0,1] = target_corr[1,0] = corr_alpha
target_corr[2,3] = target_corr[3,2] = corr_beta
target_corr[4,5] = target_corr[5,4] = corr_gamma

# Between pairs correlations
# alpha, beta - sources 0, 2 and 1, 3 are correlated
target_corr[0,2] = target_corr[2,0] = corr_ab
target_corr[1,3] = target_corr[3,1] = corr_ab
# alpha, gamma - sources 0, 4 and 1, 5 are correlated
target_corr[0,4] = target_corr[4,0] = corr_ag
target_corr[1,5] = target_corr[5,1] = corr_ag
# beta, gamma - sources 2, 4 and 3, 5 are correlated
target_corr[2,4] = target_corr[4,2] = corr_bg
target_corr[3,5] = target_corr[5,3] = corr_bg

print('Target correlation matrix:\n')
print(target_corr)

# Generate signals 
res = gen_corr_src_set(n, m,
        target_corr = target_corr,
        taper = taper,
        evoked = evoked,
        quot = quot,
        nepochs = None,
        seed_sig = gen_multi_band_noise,
        fs = fs,
        bands = bands,
        rand_seed = rand_seed,
        order = order
        )

if np.allclose(np.cov(res[0]), target_corr):
    match = 'PASSED'
else:
    match = 'FAILED'

print('Verify that COVARIANCE matrix of the generated signals matches the target CORRELATION matrix: ', match)

if match == 'FAILED':
    print('Target correlations were NOT obtained. Resutling covariance:\n')
    print(np.cov(res[0]))
    exit

# Plots
fig, axs = plt.subplots(n,3)
# fig.suptitle('Example signals')
clr = ('r','r','g','g','b','b')

for what in range(3):
    if what == 0:   # Plot the seed signals
        s = gen_multi_band_noise(n, m, fs = fs, bands = bands, rand_seed = rand_seed, order = order)
        title = ('Seed signals')
    elif what == 1: # plot the evoked parts
        s = evoked
        title = ('Evoked parts')
    elif what == 2: # plot the resulting signals
        s = res[0]
        title = ('Target signals')

    for i in range(n):
        plt.axes(axs[i,what])           # Set current axes
        tmp = max(np.abs(np.max(s[i,:])), np.abs(np.min(s[i,:])))
   
     
        if np.allclose(tmp,0):
            axs[i,what].plot(t, s[i,:], clr[i], linewidth = 1)
        else:
            axs[i,what].plot(t, s[i,:]/tmp, clr[i], linewidth = 1)
    
        plt.ylim([-1.1,1.1])
        plt.grid(True)
    
        if i == 0:
            plt.title(title)

        if i != (n-1):
            plt.tick_params(
            axis='x',           # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom=False,       # ticks along the bottom edge are off
            top=False,          # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        else:
            plt.xlabel('t, sec', loc = 'right')

fig.set_size_inches(11., 8.5)       # Letter - landscape
fig.savefig(pngname, dpi=dpi_res)
plt.show()

