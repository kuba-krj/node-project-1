import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint_adjoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_decimal_digit_label(x):
    y = torch.floor(10.0 * torch.abs(x)).long()
    return y.squeeze(-1)


def make_dataset(n_samples, device="cpu"):
    x = 2.0 * torch.rand(n_samples, 1, device=device) - 1.0
    y = first_decimal_digit_label(x)
    return x, y


class ODEFunc(nn.Module):
    def __init__(self, state_dim, hidden_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, z):
        t_feature = torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype) * t
        return self.net(torch.cat([z, t_feature], dim=1))


class NeuralODEClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1,
        latent_dim=2,
        hidden_dim=96,
        encoder_type="parabola",
        integration_end=1.0,
        decoder_mlp=False,
        ode_method="dopri5",
        ode_rtol=1e-4,
        ode_atol=1e-4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.decoder_mlp = decoder_mlp
        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        self.func = ODEFunc(state_dim=latent_dim, hidden_dim=hidden_dim)
        if decoder_mlp:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 10),
            )
        else:
            self.decoder = nn.Linear(latent_dim, 10)
        self.register_buffer(
            "integration_time",
            torch.tensor([0.0, integration_end], dtype=torch.float32),
        )

    def encoder(self, x):
        if self.encoder_type == "parabola":
            return torch.cat([x, x.square()], dim=1)
        elif self.encoder_type == "complement_zeros":
            pad = torch.zeros(
                x.shape[0],
                self.latent_dim - 1,
                device=x.device,
                dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=1)

    def solve_ode(self, z0, ts):
        return odeint_adjoint(
            self.func,
            z0,
            ts,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
            method=self.ode_method,
        )

    def forward(self, x):
        z0 = self.encoder(x)
        zt = self.solve_ode(z0, self.integration_time)
        return self.decoder(zt[-1])

    def trajectories(self, x, ts):
        z0 = self.encoder(x)
        return self.solve_ode(z0, ts)


def accuracy_from_logits(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


def evaluate_model(model, x, y, batch_size=2048):
    model.eval()
    dataloader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_size_actual = xb.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_acc += accuracy_from_logits(logits, yb) * batch_size_actual
            total_n += batch_size_actual

    return total_loss / total_n, total_acc / total_n


def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=120,
    batch_size=1024,
    lr=3e-3,
    weight_decay=1e-5,
    eval_batch_size=4096,
    restore_best_model=True,
    grad_clip_norm=1.0,
    use_cosine_scheduler=True,
):
    batch_size = min(batch_size, x_train.shape[0])
    dataloader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_cosine_scheduler
        else None
    )
    criterion = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_state = None
    best_epoch = 0
    best_val_acc = float("-inf")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_n = 0

        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_size_actual = xb.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_acc += accuracy_from_logits(logits.detach(), yb) * batch_size_actual
            total_n += batch_size_actual

        train_loss = total_loss / total_n
        train_acc = total_acc / total_n
        val_loss, val_acc = evaluate_model(
            model,
            x_val,
            y_val,
            batch_size=eval_batch_size,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step()

        improved = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_loss = val_loss

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

    if restore_best_model and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best validation checkpoint from epoch {best_epoch}.")

    return history
