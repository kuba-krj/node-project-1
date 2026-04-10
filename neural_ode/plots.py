import numpy as np
import plotly.graph_objects as go
import torch
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from .core import first_decimal_digit_label

PLOT_TEMPLATE = "plotly_white"
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
SQUARE_FIGURE_SIZE = 615
WIDE_FIGURE_WIDTH = 615
CLASS_COLORS = qualitative.Plotly
SQUARE_MARGIN = dict(l=68, r=22, t=82, b=60)
WIDE_MARGIN = dict(l=75, r=22, t=82, b=60)


def _class_color(label):
    return CLASS_COLORS[int(label) % len(CLASS_COLORS)]


def _class_colorscale(n_classes=10):
    colorscale = []
    for class_label in range(n_classes):
        left = class_label / n_classes
        right = (class_label + 1) / n_classes
        color = _class_color(class_label)
        colorscale.append((left, color))
        colorscale.append((right, color))
    return colorscale


def _single_row_figure_height(
    n_cols=1,
    horizontal_spacing=0.0,
    width=WIDE_FIGURE_WIDTH,
    margin=WIDE_MARGIN,
):
    plot_width = width - margin["l"] - margin["r"]
    subplot_width = plot_width * (1 - horizontal_spacing * (n_cols - 1)) / n_cols
    return int(round(subplot_width / GOLDEN_RATIO + margin["t"] + margin["b"]))


def _stacked_figure_height(
    n_rows,
    vertical_spacing=0.0,
    width=WIDE_FIGURE_WIDTH,
    margin=WIDE_MARGIN,
):
    plot_width = width - margin["l"] - margin["r"]
    subplot_height = plot_width / GOLDEN_RATIO
    content_height = subplot_height * n_rows / (1 - vertical_spacing * (n_rows - 1))
    return int(round(content_height + margin["t"] + margin["b"]))


def plot_training_history(result, title=None):
    title = title or f"N = {result.n_train}"
    histories = result.histories()
    show_legend = len(histories) > 1
    horizontal_spacing = 0.12

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{title} - val acc", f"{title} - val loss"),
        horizontal_spacing=horizontal_spacing,
    )

    for label, history in histories.items():
        epochs = np.arange(1, len(history["val_acc"]) + 1)
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_acc"],
                mode="lines",
                name=label,
                legendgroup=label,
                showlegend=show_legend,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_loss"],
                mode="lines",
                name=label,
                legendgroup=label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Validation accuracy", row=1, col=1, range=[0.0, 1.0])
    fig.update_yaxes(title_text="Validation loss", row=1, col=2)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        width=WIDE_FIGURE_WIDTH,
        height=_single_row_figure_height(
            n_cols=2,
            horizontal_spacing=horizontal_spacing,
        ),
        margin=WIDE_MARGIN,
    )
    fig.show()


def plot_predictions_1d(result, device="cpu"):
    grid = torch.linspace(-0.999, 0.999, 4000, device=device).unsqueeze(1)
    y_true = first_decimal_digit_label(grid).cpu().numpy()
    x_values = grid.squeeze(-1).cpu().numpy()

    models = result.models()
    n_plots = len(models)
    vertical_spacing = 0.08
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        shared_xaxes=True,
        subplot_titles=tuple(models.keys()),
        vertical_spacing=vertical_spacing,
    )

    for index, (name, model) in enumerate(models.items(), start=1):
        model.eval()
        with torch.no_grad():
            logits = model(grid)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_true,
                mode="lines",
                name="true",
                legendgroup="true",
                showlegend=index == 1,
                line=dict(width=1),
            ),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=pred,
                mode="lines",
                name="pred",
                legendgroup="pred",
                showlegend=index == 1,
                line=dict(width=1),
            ),
            row=index,
            col=1,
        )
        fig.update_yaxes(
            title_text="class",
            row=index,
            col=1,
            range=[-0.5, 9.5],
            tickmode="array",
            tickvals=list(range(10)),
        )

    fig.update_xaxes(title_text="x", row=n_plots, col=1)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        width=WIDE_FIGURE_WIDTH,
        height=_stacked_figure_height(
            n_rows=n_plots,
            vertical_spacing=vertical_spacing,
        ),
        hovermode="x unified",
        margin=WIDE_MARGIN,
    )
    fig.show()


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
    y_true = first_decimal_digit_label(xs).cpu().numpy()

    fig = go.Figure()
    seen_labels = set()
    for index in range(n_points):
        class_label = int(y_true[index])
        legend_name = f"class {class_label}"
        color = _class_color(class_label)
        showlegend = legend_name not in seen_labels
        seen_labels.add(legend_name)

        fig.add_trace(
            go.Scatter(
                x=zt[:, index, 0],
                y=zt[:, index, 1],
                mode="lines",
                name=legend_name,
                legendgroup=legend_name,
                showlegend=showlegend,
                line=dict(color=color, width=2),
                opacity=0.9,
                hovertemplate=(
                    f"target class {class_label}<br>"
                    "z1=%{x:.3f}<br>"
                    "z2=%{y:.3f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[zt[0, index, 0]],
                y=[zt[0, index, 1]],
                mode="markers",
                marker=dict(color=color, size=8, symbol="circle"),
                legendgroup=legend_name,
                showlegend=False,
                hovertemplate=(
                    f"start, class {class_label}<br>"
                    "z1=%{x:.3f}<br>"
                    "z2=%{y:.3f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[zt[-1, index, 0]],
                y=[zt[-1, index, 1]],
                mode="markers",
                marker=dict(color=color, size=9, symbol="x"),
                legendgroup=legend_name,
                showlegend=False,
                hovertemplate=(
                    f"end, class {class_label}<br>"
                    "z1=%{x:.3f}<br>"
                    "z2=%{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Trajectories in 2D latent space: {model_result.name}",
        xaxis_title="z1",
        yaxis_title="z2",
        legend_title_text="Target class",
        template=PLOT_TEMPLATE,
        width=SQUARE_FIGURE_SIZE,
        height=SQUARE_FIGURE_SIZE,
        margin=SQUARE_MARGIN,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.15)")
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.15)",
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()


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

    fig = go.Figure()
    for class_label in range(10):
        mask = y_true == class_label
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter(
                x=z_final[mask, 0],
                y=z_final[mask, 1],
                mode="markers",
                name=f"class {class_label}",
                marker=dict(
                    color=_class_color(class_label),
                    size=5,
                    opacity=0.75,
                ),
                hovertemplate=(
                    f"class {class_label}<br>"
                    "z1(T)=%{x:.3f}<br>"
                    "z2(T)=%{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Final latent representation z(T): {model_result.name}",
        xaxis_title="z1(T)",
        yaxis_title="z2(T)",
        legend_title_text="Target class",
        template=PLOT_TEMPLATE,
        width=SQUARE_FIGURE_SIZE,
        height=SQUARE_FIGURE_SIZE,
        margin=SQUARE_MARGIN,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.15)")
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.15)",
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()


def plot_head_predictions_2d(
    model_result,
    device="cpu",
    grid_limit=50.0,
    grid_points=301,
):
    model = model_result.model
    if model.latent_dim < 2:
        raise ValueError("This plot is only available for latent_dim >= 2.")

    axis = torch.linspace(-grid_limit, grid_limit, grid_points, device=device)
    z2_grid, z1_grid = torch.meshgrid(axis, axis, indexing="ij")
    latent_grid = torch.zeros(
        grid_points * grid_points,
        model.latent_dim,
        device=device,
    )
    latent_grid[:, 0] = z1_grid.reshape(-1)
    latent_grid[:, 1] = z2_grid.reshape(-1)

    model.eval()
    with torch.no_grad():
        logits = model.decoder(latent_grid)
        pred = torch.argmax(logits, dim=1).reshape(grid_points, grid_points).cpu().numpy()

    axis_values = axis.cpu().numpy()
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=axis_values,
                y=axis_values,
                z=pred,
                zmin=-0.5,
                zmax=9.5,
                colorscale=_class_colorscale(),
                showscale=True,
                colorbar=dict(
                    title="Predicted class",
                    tickmode="array",
                    tickvals=list(range(10)),
                    ticktext=[str(class_label) for class_label in range(10)],
                ),
                hovertemplate=(
                    "z1=%{x:.2f}<br>"
                    "z2=%{y:.2f}<br>"
                    "predicted class=%{z}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Head predictions on latent grid: {model_result.name}",
        xaxis_title="z1",
        yaxis_title="z2",
        template=PLOT_TEMPLATE,
        width=SQUARE_FIGURE_SIZE,
        height=SQUARE_FIGURE_SIZE,
        margin=SQUARE_MARGIN,
    )
    fig.update_xaxes(range=[-grid_limit, grid_limit], showgrid=False)
    fig.update_yaxes(
        range=[-grid_limit, grid_limit],
        showgrid=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()


def plot_experiment_report(
    result,
    device="cpu",
    augmented_model_name=None,
    latent_trajectory_points=17,
    head_grid_limit=50.0,
    head_grid_points=301,
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
    plot_head_predictions_2d(
        augmented_result,
        device=device,
        grid_limit=head_grid_limit,
        grid_points=head_grid_points,
    )


def summary_barplot(results, model_name=None):
    ns = [result.n_train for result in results]

    if model_name is None:
        model_name = results[0].model_results[0].name

    test_acc = [result.get_model_result(model_name).test_acc for result in results]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[str(n) for n in ns],
                y=test_acc,
                name=model_name,
            )
        ]
    )
    fig.update_layout(
        title="Neural ODE accuracy by training set size",
        xaxis_title="Training set size",
        yaxis_title="Test accuracy",
        yaxis_range=[0.0, 1.0],
        template=PLOT_TEMPLATE,
        width=SQUARE_FIGURE_SIZE,
        height=SQUARE_FIGURE_SIZE,
        margin=SQUARE_MARGIN,
    )
    fig.show()
