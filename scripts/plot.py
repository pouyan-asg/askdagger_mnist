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
    u_normalization = True
    impute = True
    batch_size = 128
    update_every = 5
    log_window = 1000
    s_des_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if args.s_des == "all" else [float(args.s_des)]

    fig, axs = plt.subplots(3, 3, figsize=(5.0, 0.8 * 6.50127))
    # create rainbow color map
    cmap = sns.color_palette("rainbow", 9)
    tp = np.zeros((9, reps))
    fp = np.zeros((9, reps))
    tn = np.zeros((9, reps))
    fn = np.zeros((9, reps))

    for m_i, mode in enumerate(["sensitivity", "specificity", "success"]):
        axs[0, m_i].set_ylim(0, 1)
        axs[1, m_i].set_ylim(0, 1)
        axs[2, m_i].set_ylim(0, 1)
        axs[0, m_i].set_title(f"{mode.capitalize()}-Aware Gating")
        axs[1, m_i].set_ylabel("Query Rate")
        axs[2, m_i].set_ylabel("Novice Success Rate")
        if mode == "sensitivity":
            axs[0, m_i].set_ylabel("Sensitivity")
        elif mode == "specificity":
            axs[0, m_i].set_ylabel("Specificity")
        elif mode == "success":
            axs[0, m_i].set_ylabel("System Success Rate")
        for s_i, s_des in enumerate(s_des_list):
            sens = np.zeros((reps, (60_000 // batch_size)))
            spec = np.zeros((reps, (60_000 // batch_size)))
            system_success = np.zeros((reps, (60_000 // batch_size)))
            novice_success = np.zeros((reps, (60_000 // batch_size)))
            query_rate = np.zeros((reps, (60_000 // batch_size)))
            train_samples = np.zeros((reps, (60_000 // batch_size)))

            for i in range(reps):
                dir_name = f"r{p_rand}_s{s_des}_u{u_normalization}_i{impute}_b{batch_size}_e{update_every}"
                save_path = Path(f"results_{mode}") / dir_name / f"{i}"
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
                    tp[s_i, i], fp[s_i, i], tn[s_i, i], fn[s_i, i] = update_positives_negatives(
                        tp[s_i, i], fp[s_i, i], tn[s_i, i], fn[s_i, i], q_batch, c_batch
                    )
                    sensitivity, specificity, f_window, s_window = calculate_sens_spec(
                        f_window, s_window, q_batch, c_batch, window=log_window
                    )
                    sens[i, j] = sensitivity
                    spec[i, j] = specificity
                    t_samples += np.sum(q_batch)
                    train_samples[i, j] = t_samples
                    q_batch = q[max((j + 1) * batch_size - log_window, 0) : (j + 1) * batch_size]
                    c_batch = c[max((j + 1) * batch_size - log_window, 0) : (j + 1) * batch_size]
                    window_len = len(c_batch)
                    system_success[i, j] = np.sum(np.logical_or(c_batch, q_batch)) / window_len
                    novice_success[i, j] = np.sum(c_batch) / window_len
                    query_rate[i, j] = np.sum(q_batch) / window_len

            sens_mean = np.nanmean(sens, axis=0)
            sens_std = np.nanstd(sens, axis=0)

            spec_mean = np.nanmean(spec, axis=0)
            spec_std = np.nanstd(spec, axis=0)

            system_success_mean = np.nanmean(system_success, axis=0)
            system_success_std = np.nanstd(system_success, axis=0)

            query_rate_mean = np.nanmean(query_rate, axis=0)
            query_rate_std = np.nanstd(query_rate, axis=0)

            novice_success_mean = np.nanmean(novice_success, axis=0)
            novice_success_std = np.nanstd(novice_success, axis=0)

            x = np.arange(sens.shape[1])

            if mode == "sensitivity":
                axs[0, m_i].plot(x, np.ones_like(x) * s_des, "--", color="k", alpha=0.5)
                axs[0, m_i].plot(x, sens_mean, label=r"$\sigma_\mathrm{{des}}={}$".format(s_des), color=cmap[s_i])
                axs[0, m_i].fill_between(x, sens_mean - sens_std, sens_mean + sens_std, alpha=0.3, color=cmap[s_i])
            elif mode == "specificity":
                axs[0, m_i].plot(x, np.ones_like(x) * s_des, "--", color="k", alpha=0.5)
                axs[0, m_i].plot(x, spec_mean, color=cmap[s_i])
                axs[0, m_i].fill_between(x, spec_mean - spec_std, spec_mean + spec_std, alpha=0.3, color=cmap[s_i])
            elif mode == "success":
                axs[0, m_i].plot(x, np.clip(novice_success_mean, a_min=s_des, a_max=1.0), "--", color="k", alpha=0.5)
                axs[0, m_i].plot(x, system_success_mean, color=cmap[s_i])
                axs[0, m_i].fill_between(
                    x,
                    system_success_mean - system_success_std,
                    system_success_mean + system_success_std,
                    alpha=0.3,
                    color=cmap[s_i],
                )

            axs[1, m_i].plot(x, query_rate_mean, color=cmap[s_i])
            axs[1, m_i].fill_between(
                x, query_rate_mean - query_rate_std, query_rate_mean + query_rate_std, alpha=0.3, color=cmap[s_i]
            )

            axs[2, m_i].plot(x, novice_success_mean, color=cmap[s_i])
            axs[2, m_i].fill_between(
                x,
                novice_success_mean - novice_success_std,
                novice_success_mean + novice_success_std,
                alpha=0.3,
                color=cmap[s_i],
            )

    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--"))
    labels.append(r"Desired (top row)")
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0))
    labels = [None, None, None, None, None, None, r"\textbf{A}", r"\textbf{B}", r"\textbf{C}"]
    i = 0
    for ax, label in zip(axs.flatten(), labels):
        ax.text(-0.2, -0.2, label, fontsize=8, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
        if i > 6:
            ax.plot([-0.35, -0.35], [-0.3, 6], "-", color="0.8", transform=ax.transAxes, clip_on=False)
        i += 1
    axs[2, 0].set_xlabel("Step")
    axs[2, 1].set_xlabel("Step")
    axs[2, 2].set_xlabel("Step")
    fig.set_size_inches(6.50127, 0.8 * 6.50127)
    plt.show()
    fig.savefig("figures/mnist.pdf")
