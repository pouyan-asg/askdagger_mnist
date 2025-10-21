import torch
import wandb
import numpy as np
from torch import nn
from pathlib import Path

from torch_uncertainty import TUTrainer
from torch_uncertainty.models.classification.lenet import lenet
from torch_uncertainty.models import mc_dropout
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import ClassificationRoutine

import torchvision.transforms as T

from askdagger_mnist.datamodule import CustomMNISTDataModule
from askdagger_mnist.sag import sag
from askdagger_mnist.utils import sample, evaluate, update_positives_negatives, calculate_sens_spec

torch.set_float32_matmul_precision("medium")


def train(
    iteration=0,
    p_rand=0.1,
    s_des=0.9,
    u_normalization=True,
    impute=True,
    batch_size=128,
    log=True,
    update_every=5,
    save=True,
    overwrite=False,
    verbose=True,
    accelerator="gpu",
    mode="sensitivity",
):
    assert mode in ["sensitivity", "success", "specificity"]

    # Set directory name and create save path
    name = f"r{p_rand}_s{s_des}_u{u_normalization}_i{impute}_b{batch_size}_e{update_every}"
    save_path = Path(f"results_{mode}") / name / f"{iteration}"
    save_path.mkdir(parents=True, exist_ok=True)

    if save and not overwrite and (save_path / "results.npy").exists():
        print(f"Skipping {name}")
        return

    logger = None
    if log:
        config = {
            "iteration": iteration,
            "p_rand": p_rand,
            "s_des": s_des,
            "u_normalization": u_normalization,
            "impute": impute,
            "batch_size": batch_size,
            "update_every": update_every,
        }
        logger = wandb.init(project="askdagger-mnist", name=f"{name}_i{iteration}", config=config)
    data_root = Path("data")
    enable_model_summary = True if verbose else False

    # Prepare datamodule for sampling new data
    sample_datamodule = CustomMNISTDataModule(root=data_root, batch_size=batch_size)
    sample_datamodule.prepare_data()
    sample_datamodule.setup()
    test_transform = T.Compose(
        [
            T.CenterCrop(28),
            T.Normalize(mean=sample_datamodule.mean, std=sample_datamodule.std),
        ]
    )
    model = lenet(
        in_channels=sample_datamodule.num_channels,
        num_classes=sample_datamodule.num_classes,
        dropout_rate=0.4,
    )

    # Prepare datamodule for training
    train_subset = []
    train_datamodule = CustomMNISTDataModule(
        root=data_root,
        batch_size=batch_size,
        subset=train_subset,
        num_workers=0,
        persistent_workers=False,
    )

    # Prepare routine
    # Wraps the base model with Monte Carlo Dropout, creating an ensemble of 
    # 16 models that use dropout at inference time (for uncertainty estimation).
    mc_model = mc_dropout(model, num_estimators=16, last_layer=False)
    # Creates a training and evaluation routine for classification tasks.
    routine = ClassificationRoutine(
        num_classes=sample_datamodule.num_classes,
        model=mc_model,
        loss=nn.CrossEntropyLoss(),
        optim_recipe=optim_cifar10_resnet18(mc_model),
        is_ensemble=True,
    )

    # Initialize variables
    unseen_idx = np.arange(60000)
    n_updates = 0
    max_steps = 60_000 // batch_size
    u = np.zeros((batch_size * max_steps))
    k = np.zeros((batch_size * max_steps))
    c = np.zeros((batch_size * max_steps))
    r = np.zeros((batch_size * max_steps))
    q = np.zeros((batch_size * max_steps))
    q_a = np.zeros((batch_size * max_steps))
    g = np.zeros((batch_size * max_steps))
    seed = -1 + iteration * 100
    tp, fp, tn, fn = 0, 0, 0, 0
    tp_a, fp_a, tn_a, fn_a = 0, 0, 0, 0
    f_window = []
    s_window = []
    f_a_window = []
    s_a_window = []

    for step in range(max_steps):
        seed += 2
        np.random.seed(seed)
        torch.manual_seed(seed)
        routine.eval()

        # Sample [batch_size] new images
        samples, labels, unseen_idx, sample_idx = sample(unseen_idx, batch_size, sample_datamodule, test_transform)

        # Evaluate the model on the new samples
        u_batch, prediction = evaluate(samples, routine)

        # Keep track of the age of the samples
        k_batch = [n_updates] * batch_size

        # Keep track whether the model was correct
        # This is privileged information, we only use it for logging purposes
        c_batch = prediction == labels

        u[step * batch_size : (step + 1) * batch_size] = u_batch
        k[step * batch_size : (step + 1) * batch_size] = k_batch
        c[step * batch_size : (step + 1) * batch_size] = c_batch

        # Calculate the threshold that satisfies the desired sensitivity
        gamma = sag(
            u[: (step + 1) * batch_size],
            r[: (step + 1) * batch_size],
            k[: (step + 1) * batch_size],
            s_des=s_des,
            p_rand=p_rand,
            u_normalization=u_normalization,
            impute=impute,
            mode=mode,
        )
        g[step * batch_size : (step + 1) * batch_size] = gamma

        # We sample queries randomly with rate [p_rand] and actively with threshold [gamma]
        q_r_batch = np.random.rand(batch_size) < p_rand
        q_a_batch = u_batch >= gamma
        q_a[step * batch_size : (step + 1) * batch_size] = q_a_batch

        # Keep track of queries (random and active) and rewards
        q_batch = np.logical_or(q_r_batch, q_a_batch)
        q[step * batch_size : (step + 1) * batch_size] = q_batch
        r_batch = c_batch * 2 - 1
        r_batch[~q_batch] = 0
        r[step * batch_size : (step + 1) * batch_size] = r_batch

        # Update the training subset with the queried samples
        train_subset.extend(list(sample_idx[q_batch]))
        train_datamodule.subset = train_subset

        # Update the model every [update_every] steps
        if step > 0 and step % update_every == 0:
            trainer = TUTrainer(
                accelerator=accelerator,
                max_epochs=update_every,
                enable_progress_bar=False,
                max_steps=update_every,
                log_every_n_steps=0,
                enable_model_summary=enable_model_summary,
                enable_checkpointing=False,
                logger=False,
            )
            routine.train()
            trainer.fit(model=routine, datamodule=train_datamodule)
            n_updates += 1
            enable_model_summary = False

        # Calculate sensitivity and specificity (including random queries)
        tp, fp, tn, fn = update_positives_negatives(tp, fp, tn, fn, q_batch, c_batch)
        sens = tp / (tp + fn) if tp + fn > 0 else 0
        spec = tn / (tn + fp) if tn + fp > 0 else 0
        sens_w, spec_w, f_window, s_window = calculate_sens_spec(f_window, s_window, q_batch, c_batch, 1000)

        # Calculate sensitivity and specificity (excluding random queries)
        tp_a, fp_a, tn_a, fn_a = update_positives_negatives(tp_a, fp_a, tn_a, fn_a, q_a_batch, c_batch)
        sens_a = tp_a / (tp_a + fn_a) if tp_a + fn_a > 0 else 0
        spec_a = tn_a / (tn_a + fp_a) if tn_a + fp_a > 0 else 0
        sens_a_w, spec_a_w, f_a_window, s_a_window = calculate_sens_spec(f_a_window, s_a_window, q_a_batch, c_batch, 1000)

        # Calculate active query rate (ignoring random queries)
        query_rate_a = np.sum(q_a_batch) / batch_size

        # Calculate total query rate (including random queries)
        query_rate_total = np.sum(q_batch) / batch_size

        # Calculate novice success rate (success if novice correct)
        novice_success_rate = np.mean(np.asarray(c_batch))

        # Calculate success rate (success if novice correct or queried)
        success_rate = np.sum(np.logical_or(q_batch, c_batch)) / batch_size

        if log:
            logger.log(
                {
                    "sensitivity": sens,
                    "sensitivity_window": sens_w,
                    "specificity": spec,
                    "specificity_window": spec_w,
                    "sensitivity_active": sens_a,
                    "sensitivity_active_window": sens_a_w,
                    "specificity_active": spec_a,
                    "specificity_active_window": spec_a_w,
                    "success_rate": success_rate,
                    "novice_success_rate": novice_success_rate,
                    "uncertainty": np.mean(u_batch),
                    "threshold": gamma,
                    "n_samples": len(train_subset),
                    "s_desired": s_des,
                    "query_rate_active": query_rate_a,
                    "query_rate": query_rate_total,
                }
            )
        if verbose:
            print(f"step: {step}")
            if mode == "sensitivity":
                print(f"sensitivity - total: {sens:.2f}, window: {sens_w:.2f} desired: {s_des:.2f}")
                print(f"succes rate - system: {success_rate:.2f}, novice: {novice_success_rate:.2f}")
            elif mode == "specificity":
                print(f"specificity - total: {spec:.2f}, window: {spec_w:.2f} desired: {s_des:.2f}")
                print(f"succes rate - system: {success_rate:.2f}, novice: {novice_success_rate:.2f}")
            elif mode == "success":
                print(f"succes rate - system: {success_rate:.2f}, minimum desired: {s_des:.2f}")
                print(f"Novice success rate: {novice_success_rate:.2f}")
            print(f"query rate: {query_rate_total:.2f}")
            print("-" * 80)

    if save:
        torch.save(routine.model.state_dict(), save_path / f"{name}.pt")
        results = {
            "u": u,
            "k": k,
            "g": g,
            "c": c,
            "r": r,
            "q": q,
            "q_a": q_a,
            "train_subset": train_subset,
        }
        np.save(save_path / "results.npy", results)
    if log:
        logger.finish()
