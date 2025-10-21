from askdagger_mnist.train import train
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--s_des", type=str, default="all", choices=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "all"]
    )
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args()

    train_kwargs = []
    s_des_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if args.s_des == "all" else [float(args.s_des)]

    for mode in ["sensitivity", "specificity", "success"]:
        for i in range(args.reps):
            for s_des in s_des_list:
                train_kwargs.append(
                    {"iteration": i, "s_des": s_des, "accelerator": args.accelerator, "overwrite": args.overwrite, "mode": mode}
                )

    print(train_kwargs)
    for i in range(len(train_kwargs)):
        train(**train_kwargs[i])
