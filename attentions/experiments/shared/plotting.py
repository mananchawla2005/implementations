import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt


def plot_times(seq_lens, times_dict, xlabel="Sequence length", ylabel="Time (s)", title=None):
    for label, times in times_dict.items():
        plt.plot(seq_lens, times, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


SEQ_LENS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 4000]
