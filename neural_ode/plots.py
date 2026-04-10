import matplotlib.pyplot as plt
import numpy as np
import torch

from .core import first_decimal_digit_label


def plot_training_history(result, title=None):
    title = title or f"N = {result.n_train}"
    histories = result.histories()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for label, history in histories.items():
        plt.plot(history["val_acc"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title(f"{title} - val acc")
    if len(histories) > 1:
        plt.legend()

    plt.subplot(1, 2, 2)
    for label, history in histories.items():
        plt.plot(history["val_loss"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title(f"{title} - val loss")
    if len(histories) > 1:
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_predictions_1d(result, device="cpu"):
    grid = torch.linspace(-0.999, 0.999, 4000, device=device).unsqueeze(1)
    y_true = first_decimal_digit_label(grid).cpu().numpy()

    plt.figure(figsize=(12, 6))

    models = result.models()
    n_plots = len(models)
    for index, (name, model) in enumerate(models.items(), start=1):
        model.eval()
        with torch.no_grad():
            logits = model(grid)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        plt.subplot(n_plots, 1, index)
        plt.plot(grid.cpu().numpy(), y_true, label="true", linewidth=1)
        plt.plot(grid.cpu().numpy(), pred, label="pred", linewidth=1)
        plt.ylim(-0.5, 9.5)
        plt.ylabel("class")
        plt.title(name)
        plt.legend()

    plt.xlabel("x")
    plt.tight_layout()
    plt.show()


def plot_latent_trajectories_2d(model_result, device="cpu", n_points=15):
    model = model_result.model
    if model.latent_dim < 2:
        raise ValueError("This plot is only available for latent_dim >= 2.")

    xs = torch.linspace(-0.95, 0.95, n_points, device=device).unsqueeze(1)
    ts = torch.linspace(
        float(model.integration_time[0].item()),
        float(model.integration_time[-1].item()),
        100,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        zt = model.trajectories(xs, ts).cpu().numpy()

    plt.figure(figsize=(7, 7))
    for index in range(n_points):
        plt.plot(zt[:, index, 0], zt[:, index, 1], alpha=0.9)
        plt.scatter(zt[0, index, 0], zt[0, index, 1], s=20, marker="o")
        plt.scatter(zt[-1, index, 0], zt[-1, index, 1], s=20, marker="x")

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(f"Trajectories in 2D latent space: {model_result.name}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_final_latent_and_classes(model_result, device="cpu"):
    model = model_result.model
    if model.latent_dim < 2:
        raise ValueError("This plot is only available for latent_dim >= 2.")

    grid = torch.linspace(-0.999, 0.999, 2000, device=device).unsqueeze(1)
    y_true = first_decimal_digit_label(grid).cpu().numpy()

    model.eval()
    with torch.no_grad():
        z_final = model.trajectories(grid, model.integration_time.to(device))[-1]
        z_final = z_final.cpu().numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_final[:, 0], z_final[:, 1], c=y_true, s=8, alpha=0.8)
    plt.xlabel("z1(T)")
    plt.ylabel("z2(T)")
    plt.title(f"Final latent representation z(T): {model_result.name}")
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, ticks=range(10))
    plt.show()


def plot_experiment_report(
    result,
    device="cpu",
    augmented_model_name=None,
    latent_trajectory_points=17,
):
    plot_training_history(result, title=f"N = {result.n_train}")
    plot_predictions_1d(result, device=device)

    if augmented_model_name is None or len(result.model_results) == 1:
        augmented_result = result.model_results[0]
    else:
        augmented_result = result.get_model_result(augmented_model_name)

    plot_latent_trajectories_2d(
        augmented_result,
        device=device,
        n_points=latent_trajectory_points,
    )
    plot_final_latent_and_classes(augmented_result, device=device)


def summary_barplot(results, model_name=None):
    ns = [result.n_train for result in results]

    if model_name is None:
        model_name = results[0].model_results[0].name

    test_acc = [result.get_model_result(model_name).test_acc for result in results]
    x = np.arange(len(ns))

    plt.figure(figsize=(8, 5))
    plt.bar(x, test_acc, width=0.6, label=model_name)
    plt.xticks(x, [str(n) for n in ns])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Training set size")
    plt.ylabel("Test accuracy")
    plt.title("Neural ODE accuracy by training set size")
    plt.legend()
    plt.show()
