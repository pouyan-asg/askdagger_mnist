import argparse
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
from askdagger_mnist.utils import calculate_sens_spec, update_positives_negatives, set_plot_style


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument(
        "--s_des", type=str, default="all", choices=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "all"]
    )
    args = parser.parse_args()

    figure_dir = Path("figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    reps = args.reps
    p_rand = 0.1
    batch_size = 128
    update_every = 5
    log_window = 1000
    s_des_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if args.s_des == "all" else [float(args.s_des)]

    fig, axs = plt.subplots(1, 3, figsize=(6.50127, 0.4 * 6.50127))
    # create rainbow color map
    cmap = sns.color_palette("rainbow", 9)
    idx = 0

    tp = np.zeros((3, 9, reps))
    fp = np.zeros((3, 9, reps))
    tn = np.zeros((3, 9, reps))
    fn = np.zeros((3, 9, reps))
    for setting in [(True, True), (True, False), (False, True)]:
        u_normalization, impute = setting
        sens = np.zeros((reps, (60_000 // batch_size)))
        for sens_i, s_des in enumerate(s_des_list):
            if u_normalization and impute:
                title = r"SAG"
            elif impute:
                title = r"SAG w/o Normalization"
            else:
                title = r"SAG w/o Imputation"

            for i in range(reps):
                dir_name = f"r{p_rand}_s{s_des}_u{u_normalization}_i{impute}_b{batch_size}_e{update_every}"
                save_path = Path("results") / dir_name / f"{i}"
                if (save_path / "results.npy").exists():
                    results = np.load(save_path / "results.npy", allow_pickle=True).item()
                else:
                    print(("-" * 80) + f"\n{dir_name} not found\n" + ("-" * 80))
                    continue
                c = np.asarray(results["c"])
                q = np.asarray(results["q"])
                f_window = []
                s_window = []
                t_samples = 0
                for j in range(sens.shape[1]):
                    # Calculate sensitivity and specificity (including random queries)
                    q_batch = q[j * batch_size : (j + 1) * batch_size]
                    c_batch = c[j * batch_size : (j + 1) * batch_size]
                    tp[idx, sens_i, i], fp[idx, sens_i, i], tn[idx, sens_i, i], fn[idx, sens_i, i] = update_positives_negatives(
                        tp[idx, sens_i, i], fp[idx, sens_i, i], tn[idx, sens_i, i], fn[idx, sens_i, i], q_batch, c_batch
                    )
                    sensitivity, specificity, f_window, s_window = calculate_sens_spec(
                        f_window, s_window, q_batch, c_batch, window=log_window
                    )
                    sens[i, j] = sensitivity

            sens_mean = np.nanmean(sens, axis=0)
            sens_std = np.nanstd(sens, axis=0)

            x = np.arange(sens.shape[1])

            axs[idx].plot(x, np.ones_like(x) * s_des, "--", color="k", alpha=0.5)
            axs[idx].plot(x, sens_mean, label=r"$\sigma_\mathrm{{des}}={}$".format(s_des), color=cmap[sens_i])
            axs[idx].fill_between(x, sens_mean - sens_std, sens_mean + sens_std, alpha=0.3, color=cmap[sens_i])
        axs[idx].set_title(title)
        axs[idx].set_xlabel("Step")
        axs[idx].set_ylim(0, 1)
        idx += 1

    axs[0].set_ylabel("Sensitivity")
    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--"))
    labels.append(r"Desired")
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0))
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}"]
    for ax, label in zip(axs.flatten(), labels):
        ax.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
    plt.show()
    fig.savefig("figures/mnist_reg_ablation.pdf")

    # Create table
    sens = tp / (tp + fn)
    sens_mean_sag = np.mean(sens[0], axis=1)
    sens_std_sag = np.std(sens[0], axis=1)
    sens_mean_wo_imp = np.mean(sens[1], axis=1)
    sens_std_wo_imp = np.std(sens[1], axis=1)
    sens_mean_wo_norm = np.mean(sens[2], axis=1)
    sens_std_wo_norm = np.std(sens[2], axis=1)
    print("$\sigma_\mathrm{des}$ & SAG & SAG w/o Imputation & SAG w/o Normalization \\\\")
    for i, s_des in enumerate(s_des_list):
        print(
            f"{s_des} & ${sens_mean_sag[i]:.3f} \pm {sens_std_sag[i]:.3f}$ & ${sens_mean_wo_imp[i]:.3f} \pm {sens_std_wo_imp[i]:.3f}$ & ${sens_mean_wo_norm[i]:.3f} \pm {sens_std_wo_norm[i]:.3f}$ \\\\"
        )
