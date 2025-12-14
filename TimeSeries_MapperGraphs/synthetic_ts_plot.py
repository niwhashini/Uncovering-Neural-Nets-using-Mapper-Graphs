# synthetic_ts_plot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Synthetic dataset generator (same as before) ----------

def generate_base_signal(T, n_channels):
    t = np.linspace(0, 50*np.pi, T)
    base = []

    for c in range(n_channels):
        freq = np.random.uniform(0.02, 0.06)
        phase = np.random.uniform(0, 2*np.pi)
        sin_part = np.sin(freq*t + phase)

        ar_noise = np.zeros(T)
        alpha = np.random.uniform(0.7, 0.95)
        eps = np.random.normal(0, 0.5, T)
        for i in range(1, T):
            ar_noise[i] = alpha * ar_noise[i-1] + eps[i]

        base.append(sin_part + 0.3 * ar_noise)

    return np.vstack(base).T

def inject_mean_shift(data, start, length):
    shift = np.random.uniform(3, 6)
    data[start:start+length] += shift
    return data

def inject_spike(data, start):
    spike_mag = np.random.uniform(10, 20)
    data[start] += spike_mag
    return data

def inject_variance_burst(data, start, length):
    noise = np.random.normal(0, 4, (length, data.shape[1]))
    data[start:start+length] += noise
    return data

def inject_dropout(data, start, length, channel):
    data[start:start+length, channel] = 0
    return data

def inject_regime_switch(data, start, length, channel):
    t = np.linspace(0, 20*np.pi, length)
    data[start:start+length, channel] = np.cos(0.1*t) * 5
    return data

def generate_synthetic_dataset(
    T=200_000, n_channels=8, seed=42,
    mean_shift=True, spikes=True, variance=True, dropout=True, regime=True
):
    np.random.seed(seed)

    data = generate_base_signal(T, n_channels)
    labels_global = np.zeros(T)
    labels_type = np.zeros(T)

    # Type 1: Mean shift
    if mean_shift:
        s, l = 20000, 8000
        data = inject_mean_shift(data, s, l)
        labels_global[s:s+l] = 1
        labels_type[s:s+l] = 1

    # Type 2: Spikes
    if spikes:
        spike_points = [40000, 80000, 120000]
        for sp in spike_points:
            data = inject_spike(data, sp)
            labels_global[sp] = 1
            labels_type[sp] = 2

    # Type 3: Variance bursts
    if variance:
        s, l = 100000, 5000
        data = inject_variance_burst(data, s, l)
        labels_global[s:s+l] = 1
        labels_type[s:s+l] = 3

    # Type 4: Dropout
    if dropout:
        s, l = 150000, 5000
        ch = np.random.randint(0, data.shape[1])
        data = inject_dropout(data, s, l, ch)
        labels_global[s:s+l] = 1
        labels_type[s:s+l] = 4

    # Type 5: Regime switch
    if regime:
        s, l = 170000, 7000
        ch = np.random.randint(0, data.shape[1])
        data = inject_regime_switch(data, s, l, ch)
        labels_global[s:s+l] = 1
        labels_type[s:s+l] = 5

    df = pd.DataFrame(data, columns=[f"ch{i}" for i in range(data.shape[1])])
    df["label"] = labels_global
    df["type"] = labels_type

    return df

# ---------- Plotting helpers ----------

ANOMALY_COLORS = {
    1: "orange",   # mean shift
    2: "red",      # spikes
    3: "purple",   # variance burst
    4: "cyan",     # dropout
    5: "green",    # regime switch
}

ANOMALY_NAMES = {
    0: "Normal",
    1: "Mean shift",
    2: "Spike",
    3: "Variance burst",
    4: "Dropout",
    5: "Regime switch",
}

def plot_multichannel_with_anomalies(df, channels_to_plot=None, T_max=5000):
    """
    Quick overview of first T_max timesteps, with anomaly regions shaded.
    """
    if channels_to_plot is None:
        channels_to_plot = [0, 1, 2, 3]  # first 4 channels

    n_channels = len(channels_to_plot)
    t = np.arange(min(T_max, len(df)))

    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels_to_plot):
        y = df[f"ch{ch}"].values[:len(t)]
        ax.plot(t, y, linewidth=0.8)
        ax.set_ylabel(f"ch{ch}")

        # Shade anomalies by type
        types = df["type"].values[:len(t)]
        for a_type, color in ANOMALY_COLORS.items():
            mask = types == a_type
            if not mask.any():
                continue
            # find contiguous segments
            idx = np.where(mask)[0]
            splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            for seg in splits:
                ax.axvspan(seg[0], seg[-1], color=color, alpha=0.2)

    axes[-1].set_xlabel("Time index")

    # Legend
    handles = [plt.Line2D([0], [0], color=c, lw=4, alpha=0.5)
               for k, c in ANOMALY_COLORS.items()]
    labels = [ANOMALY_NAMES[k] for k in ANOMALY_COLORS.keys()]
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Synthetic multichannel time series with anomaly regions", fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_anomaly_type_timeline(df, T_max=50000):
    """
    Single-line plot showing which anomaly type occurs over time.
    """
    t = np.arange(min(T_max, len(df)))
    types = df["type"].values[:len(t)]

    fig, ax = plt.subplots(figsize=(15, 2))
    ax.scatter(t, types, s=5, c=[
        ANOMALY_COLORS.get(int(tp), "gray") for tp in types
    ])
    ax.set_yticks(sorted(ANOMALY_NAMES.keys()))
    ax.set_yticklabels([ANOMALY_NAMES[k] for k in sorted(ANOMALY_NAMES.keys())])
    ax.set_xlabel("Time index")
    ax.set_title("Anomaly type over time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate and save
    df = generate_synthetic_dataset()
    df.to_csv("synthetic_multidim_ts.csv", index=False)

    print(df["type"].value_counts())

    # Quick plots
    plot_multichannel_with_anomalies(df, channels_to_plot=[0, 1, 2, 3], T_max=200000)
    plot_anomaly_type_timeline(df, T_max=200000)

