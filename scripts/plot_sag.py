import argparse
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve
from askdagger_mnist.utils import set_plot_style


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument(
        "--s_des", type=str, default="0.8", choices=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    )
    args = parser.parse_args()

    figure_dir = Path("figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    p_rand = 0.1
    u_normalization = True
    impute = True
    n_min = 50
    batch_size = 128
    update_every = 5
    log_window = 1000

    i = args.rep
    s_des = float(args.s_des)

    dir_name = f"r{p_rand}_s{s_des}_u{u_normalization}_i{impute}_b{batch_size}_e{update_every}"
    save_path = Path("results") / dir_name / f"{i}"
    if (save_path / "results.npy").exists():
        results = np.load(save_path / "results.npy", allow_pickle=True).item()
    else:
        print(("-" * 80) + f"\n{dir_name} not found\n" + ("-" * 80))
        exit
    u_array = np.asarray(results["u"])
    k_array = np.asarray(results["k"], dtype="int")
    r_array = np.asarray(results["r"])

    window_len = 0
    window_idx = k_array >= k_array[-1] - window_len
    u_window = u_array[window_idx]
    r_window = r_array[window_idx]
    k_window = k_array[window_idx]
    known = np.logical_not(r_window == 0)

    while np.sum(r_window == -1) < n_min or np.sum(r_window == 1) < 1:
        window_len += 1
        window_idx = k_array >= k_array[-1] - window_len
        u_window = u_array[window_idx]
        r_window = r_array[window_idx]
        k_window = k_array[window_idx]
        known = np.logical_not(r_window == 0)

    x = k_window
    y = u_window

    uncertainty_mean = LinearRegression().fit(x.reshape(-1, 1), y)
    u_predictions = uncertainty_mean.predict(x.reshape(-1, 1))
    u_window_normalized = u_window - uncertainty_mean.predict(x.reshape(-1, 1))
    u_window_normalized += uncertainty_mean.predict(np.array([k_window[-1]]).reshape(-1, 1))

    unknown = np.logical_not(known)
    y = -r_window[known]
    X = u_window_normalized[known].reshape(-1, 1)

    clf = LogisticRegression(penalty=None).fit(X, y)
    probas = clf.predict_proba(u_window_normalized.reshape(-1, 1))
    probas_unknown = clf.predict_proba(u_window_normalized[unknown].reshape(-1, 1))

    sorted_idx = np.argsort(u_window_normalized)
    u_window_normalized_sorted = u_window_normalized[sorted_idx]
    probas_sorted = probas[sorted_idx]

    fig, axs = plt.subplots(1, 4, figsize=(6.50127, 0.3 * 6.50127), dpi=300)

    scatter1 = axs[0].scatter(k_window[r_window == 0], u_window[r_window == 0], color="k", s=1)
    scatter2 = axs[0].scatter(k_window[r_window == 1], u_window[r_window == 1], color="g", s=1)
    scatter3 = axs[0].scatter(k_window[r_window == -1], u_window[r_window == -1], color="r", s=1)
    lin = axs[0].plot(k_window, u_predictions, color="b", linestyle="-")
    axs[0].plot(k_window, u_predictions, color="b", linestyle="-")
    axs[0].annotate(
        "",
        xy=(k_window[0] - 0.04, u_predictions[-1]),
        xytext=(k_window[0] - 0.04, u_predictions[0]),
        arrowprops=dict(arrowstyle="-", shrinkA=5, shrinkB=5, color="k", connectionstyle="bar"),
        fontsize=8,
        color="k",
        ha="center",
        va="center",
    )
    axs[0].text(
        k_window[0] - 0.42,
        (u_predictions[0] + u_predictions[-1]) / 2,
        r"$w_\mathrm{lin}$",
    )
    axs[0].set_xlabel(r"Update count $k$")
    axs[0].set_ylabel(r"Uncertainty $u$")
    axs[0].set_xticks([92, 93])
    axs[0].set_xlim([91.55, 93.2])

    axs[1].scatter(u_window_normalized[r_window == -1], -r_window[r_window == -1], color="r", s=1)
    axs[1].scatter(u_window_normalized[r_window == 0], -r_window[r_window == 0], color="k", s=1)
    axs[1].annotate(
        "",
        xy=(min(u_window_normalized[r_window == 0]), -0.2),
        xytext=(max(u_window_normalized[r_window == 0]), -0.2),
        arrowprops=dict(arrowstyle="-", shrinkA=10, shrinkB=10, color="k", connectionstyle="bar"),
        fontsize=8,
        color="k",
        ha="center",
        va="center",
    )
    axs[1].text(
        (min(u_window_normalized[r_window == 0]) + max(u_window_normalized[r_window == 0])) / 2,
        0.22,
        r"\textnormal{Not queried}",
        ha="center",
        va="bottom",
    )
    axs[1].scatter(u_window_normalized[r_window == 1], -r_window[r_window == 1], color="g", s=1)
    axs[1].set_xlabel(r"Uncertainty $u$")
    axs[1].set_ylabel(r"Negated reward $-r$")
    axs[1].set_yticks([-1, 0, 1])

    p1 = axs[2].plot(u_window_normalized_sorted, probas_sorted[:, 0], color="g")
    p2 = axs[2].plot(u_window_normalized_sorted, probas_sorted[:, 1], color="r")
    axs[2].text(
        u_window_normalized_sorted[0] + 0.02,
        0.5,
        r"\textnormal{Fit on data in} \textbf{B}",
        ha="left",
        va="center",
    )
    axs[2].set_xlabel(r"Uncertainty $u$")
    axs[2].set_ylabel("Probability $P$")

    failures = -r_window.copy()
    failures[unknown] = np.asarray(probas_unknown[:, 1] > np.random.rand(probas_unknown.shape[0]), dtype="int") * 2 - 1
    _, tpr, threshs = roc_curve(failures, u_window_normalized, pos_label=1)
    fnr = 1 - tpr
    gamma = np.interp(s_des, tpr + p_rand * fnr, threshs)

    imputed_failures = np.logical_and(unknown, failures == 1)
    imputed_successes = np.logical_and(unknown, failures == -1)

    axs[3].scatter(u_window_normalized[imputed_failures], failures[imputed_failures], color="k", label=r"$r = 0$", s=1)
    axs[3].annotate(
        "",
        xy=(min(u_window_normalized[imputed_failures]), 1),
        xytext=(min(u_window_normalized[imputed_failures]), 0.5),
        arrowprops=dict(arrowstyle="->", shrinkA=5, shrinkB=5, color="k"),
        fontsize=8,
        color="k",
        ha="center",
        va="center",
    )
    axs[3].text(
        min(u_window_normalized[imputed_failures]) + 0.02,
        0.6,
        r"$\sim P(f|u)$",
        ha="center",
        va="top",
    )
    axs[3].scatter(u_window_normalized[imputed_successes], failures[imputed_successes], color="k", label=r"$r = 0$", s=1)
    axs[3].scatter(u_window_normalized[r_window == 1], -r_window[r_window == 1], color="g", label=r"$r = 1$", s=1)
    axs[3].scatter(u_window_normalized[r_window == -1], -r_window[r_window == -1], color="r", label=r"$r = -1$", s=1)
    g = axs[3].axvline(gamma, color="k", linestyle="--")
    axs[3].set_xlabel(r"Uncertainty $u$")
    axs[3].set_ylabel(r"Failure $f$")
    axs[3].set_yticks([-1, 0, 1])

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    handles = [scatter2, scatter1, scatter3, lin[0], p1[0], p2[0], g]
    labels = [r"$r = 1$", r"$r = 0$", r"$r = -1$", r"\texttt{LinRegres}", r"$P(f=-1|u)$", r"$P(f=1|u)$", r"$\gamma_i$"]
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.1), ncol=7)

    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}"]
    for ax, label in zip(axs.flatten(), labels):
        ax.grid(False)
        ax.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
    plt.show()
    fig.savefig("figures/sag.pdf")
