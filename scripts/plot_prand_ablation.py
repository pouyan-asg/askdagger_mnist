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
    batch_size = 128
    update_every = 5
    log_window = 1000
    u_normalization = True
    impute = True
    s_des_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if args.s_des == "all" else [float(args.s_des)]

    fig, axs = plt.subplots(2, 3, figsize=(6.50127, 0.7 * 6.50127))
    # create rainbow color map
    cmap = sns.color_palette("rainbow", 9)
    idx = 0

    tp = np.zeros((3, 9, reps))
    fp = np.zeros((3, 9, reps))
    tn = np.zeros((3, 9, reps))
    fn = np.zeros((3, 9, reps))
    for p_rand in [0.05, 0.1, 0.2]:
        sens = np.zeros((reps, (60_000 // batch_size)))
        spec = np.zeros((reps, (60_000 // batch_size)))
        for sens_i, s_des in enumerate(s_des_list):
            title = rf"SAG $p_\mathrm{{rand}}$={p_rand}"

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
                    spec[i, j] = specificity

            sens_mean = np.nanmean(sens, axis=0)
            sens_std = np.nanstd(sens, axis=0)

            spec_mean = np.nanmean(spec, axis=0)
            spec_std = np.nanstd(spec, axis=0)

            x = np.arange(sens.shape[1])

            axs[0, idx].plot(x, np.ones_like(x) * s_des, "--", color="k", alpha=0.5)
            axs[0, idx].plot(x, sens_mean, label=r"$\sigma_\mathrm{{des}}={}$".format(s_des), color=cmap[sens_i])
            axs[0, idx].fill_between(x, sens_mean - sens_std, sens_mean + sens_std, alpha=0.3, color=cmap[sens_i])

            axs[1, idx].plot(x, spec_mean, color=cmap[sens_i])
            axs[1, idx].fill_between(x, spec_mean - spec_std, spec_mean + spec_std, alpha=0.3, color=cmap[sens_i])
        axs[0, idx].set_title(title)
        axs[1, idx].set_xlabel("Step")
        axs[0, idx].set_ylim(0, 1)
        axs[1, idx].set_ylim(0, 1)
        idx += 1

    axs[0, 0].set_ylabel("Sensitivity")
    axs[1, 0].set_ylabel("Specificity")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--"))
    labels.append(r"Desired \bf{(A-C)}")
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0))
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}", r"\textbf{E}", r"\textbf{F}"]
    for ax, label in zip(axs.flatten(), labels):
        ax.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
    plt.show()
    fig.savefig("figures/mnist_prand_ablation.pdf")

    # Create tables
    sens = tp / (tp + fn)
    sens_mean_05 = np.mean(sens[0], axis=1)
    sens_std_05 = np.std(sens[0], axis=1)
    sens_mean_1 = np.mean(sens[1], axis=1)
    sens_std_1 = np.std(sens[1], axis=1)
    sens_mean_2 = np.mean(sens[2], axis=1)
    sens_std_2 = np.std(sens[2], axis=1)

    print(
        "$\sigma_\mathrm{des}$ & sensitivity $p_\mathrm{rand}=0.05$ & sensitivity $p_\mathrm{rand}=0.1$ & sensitivity $p_\mathrm{rand}=0.2$ \\\\"
    )
    for i, s_des in enumerate(s_des_list):
        print(
            f"{s_des} & ${sens_mean_05[i]:.3f} \pm {sens_std_05[i]:.3f}$ & ${sens_mean_1[i]:.3f} \pm {sens_std_1[i]:.3f}$ & ${sens_mean_2[i]:.3f} \pm {sens_std_2[i]:.3f}$ \\\\"
        )

    spec = tn / (tn + fp)
    inform_mean_05 = np.mean(sens[0] + spec[0] - 1, axis=1)
    inform_std_05 = np.std(sens[0] + spec[0] - 1, axis=1)
    inform_mean_1 = np.mean(sens[1] + spec[1] - 1, axis=1)
    inform_std_1 = np.std(sens[1] + spec[1] - 1, axis=1)
    inform_mean_2 = np.mean(sens[2] + spec[2] - 1, axis=1)
    inform_std_2 = np.std(sens[2] + spec[2] - 1, axis=1)

    print(
        "$\sigma_\mathrm{des}$ & informedness $p_\mathrm{rand}=0.05$ & informedness $p_\mathrm{rand}=0.1$ & informedness $p_\mathrm{rand}=0.2$ \\\\"
    )
    for i, s_des in enumerate(s_des_list):
        print(
            f"{s_des} & ${inform_mean_05[i]:.3f} \pm {inform_std_05[i]:.3f}$ & ${inform_mean_1[i]:.3f} \pm {inform_std_1[i]:.3f}$ & ${inform_mean_2[i]:.3f} \pm {inform_std_2[i]:.3f}$ \\\\"
        )
