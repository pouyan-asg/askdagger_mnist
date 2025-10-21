import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def set_plot_style():
    sns.set_theme(style="whitegrid", font_scale=1)
    plt.rcParams.update(
        {
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "font.size": 8,
            "grid.linewidth": 0.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.pad": -2.0,
            "ytick.major.pad": -2.0,
            "lines.linewidth": 1.3,
            "axes.xmargin": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": True,
            "font.family": "Helvetica",
        }
    )


def sample(data_idx, batch_size, datamodule, transform):
    sample_idx = np.random.choice(data_idx, batch_size, replace=False)
    tobe_removed = np.where(np.isin(data_idx, sample_idx))[0]
    data_idx = np.delete(data_idx, tobe_removed)
    samples = datamodule.train.data[sample_idx]
    labels = datamodule.train.targets[sample_idx]
    samples = transform(samples / 255).unsqueeze(1)
    return samples, labels.detach().cpu().numpy(), data_idx, sample_idx


def evaluate(samples, routine):
    logits = routine(samples).reshape(16, -1, 10)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    values, predicted = torch.max(torch.mean(probs, dim=0), 1)
    uncertainty = 1 - values
    prediction = torch.argmax(probs.mean(dim=0), dim=-1)
    uncertainty[torch.isnan(uncertainty)] = 1
    return uncertainty.detach().cpu().numpy(), prediction.detach().cpu().numpy()


def update_positives_negatives(tp, fp, tn, fn, q, c):
    tn += np.sum(np.logical_and(np.logical_not(q), np.asarray(c)))
    tp += np.sum(np.logical_and(q, np.logical_not(np.asarray(c))))
    fn += np.sum(np.logical_and(np.logical_not(q), np.logical_not(np.asarray(c))))
    fp += np.sum(np.logical_and(q, np.asarray(c)))
    return tp, fp, tn, fn


def calculate_sens_spec(failure_window, success_window, q, c, window):
    failures = np.where(np.asarray(c) == 0)[0]
    successes = np.where(np.asarray(c) == 1)[0]
    failure_window.extend(q[failures])
    success_window.extend(q[successes])
    failure_window = failure_window[-window:]
    success_window = success_window[-window:]

    sens = np.sum(failure_window) / len(failure_window)
    spec = 1 - np.sum(success_window) / len(success_window)
    return sens, spec, failure_window, success_window
