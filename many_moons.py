from __future__ import annotations

import pathlib
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons
from torch import nn


class TwoLabelMoonsDataset:
    def __init__(self, n_samples=10_000, noise=0.05, dtype=torch.float32):
        data, labels = make_moons(n_samples, noise=noise)
        self.data = torch.tensor(data, dtype=dtype)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def sample(self, batch_size):
        idx = torch.randint(0, self.data.shape[0], (batch_size,))
        return self.data[idx], self.labels[idx]


class ConditionalVectorField(nn.Module):
    def __init__(self, hidden_size=128, embedding_dim=4):
        super().__init__()
        self.label_emb = nn.Embedding(2, embedding_dim)
        in_features = 2 + 1 + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x_t, t, labels):
        if labels.dim() == 2:
            labels = labels.squeeze(-1)
        emb = self.label_emb(labels)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x_t, t, emb], dim=-1))


def flow_matching_loss(model, data_batch, labels):
    noise = torch.randn_like(data_batch)
    t = torch.rand(data_batch.shape[0], 1)
    x_t = (1 - t) * noise + t * data_batch
    target_velocity = data_batch - noise
    predicted_velocity = model(x_t, t, labels)
    return torch.mean((predicted_velocity - target_velocity) ** 2)


@torch.no_grad()
def sample_from_flow(model, label, num_samples=2048, integration_steps=80):
    model.eval()
    x = torch.randn(num_samples, 2)
    y = torch.full((num_samples,), label, dtype=torch.long)
    dt = 1 / integration_steps
    for step in range(integration_steps):
        t = torch.full((num_samples, 1), step / integration_steps)
        velocity = model(x, t, y)
        x = x + dt * velocity
    return x


@torch.no_grad()
def sample_flow_trajectory(model, label, num_samples=1024, integration_steps=80):
    model.eval()
    x = torch.randn(num_samples, 2)
    y = torch.full((num_samples,), label, dtype=torch.long)
    states = [x.clone()]
    dt = 1 / integration_steps
    for step in range(integration_steps):
        t = torch.full((num_samples, 1), step / integration_steps)
        velocity = model(x, t, y)
        x = x + dt * velocity
        states.append(x.clone())
    return torch.stack(states)


def save_plot(dataset, samples_per_label, out_path):
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.labels, cmap="coolwarm", s=5, alpha=0.5)
    axes[1].scatter(samples_per_label[0][:, 0], samples_per_label[0][:, 1], c="#1f77b4", s=5, alpha=0.6)
    axes[2].scatter(samples_per_label[1][:, 0], samples_per_label[1][:, 1], c="#d62728", s=5, alpha=0.6)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-2, 3)
        ax.set_ylim(-1.5, 1.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_trajectory_plot(trajectories, out_path, panels=5, points_per_panel=500):
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(trajectories.keys())
    fig, axes = plt.subplots(len(labels), panels, figsize=(3 * panels, 3 * len(labels)))
    if len(labels) == 1:
        axes = axes[None, :]
    for row, label in enumerate(labels):
        traj = trajectories[label]
        num_steps = traj.shape[0] - 1
        step_indices = torch.round(torch.linspace(0, num_steps, steps=panels)).long()
        for col, step_idx in enumerate(step_indices):
            pts = traj[step_idx]
            if pts.shape[0] > points_per_panel:
                idx = torch.randperm(pts.shape[0])[:points_per_panel]
                pts = pts[idx]
            ax = axes[row, col]
            ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.6, c="#1f77b4" if label == 0 else "#d62728")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-2, 3)
            ax.set_ylim(-1.5, 1.5)
            ax.set_title(f"label {label} | t={step_idx / num_steps:.2f}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(0)

    data_size = 10_000
    data_noise = 0.05
    hidden_size = 128
    train_steps = 3000
    batch_size = 512
    lr = 3e-4
    log_every = 200

    out_plot = "moons_flow_matching.png"
    trajectory_plot = "moons_flow_matching_trajectory.png"
    trajectory_panels = 5
    trajectory_samples = 1024

    dataset = TwoLabelMoonsDataset(n_samples=data_size, noise=data_noise)
    model = ConditionalVectorField(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, train_steps + 1):
        batch, labels = dataset.sample(batch_size)
        loss = flow_matching_loss(model, batch, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == 1:
            print(f"step {step:05d} | loss {loss.item():.4f}")

    samples = {label: sample_from_flow(model, label) for label in (0, 1)}
    save_plot(dataset, samples, out_plot)

    trajectories = {
        label: sample_flow_trajectory(model, label, num_samples=trajectory_samples)
        for label in (0, 1)
    }
    save_trajectory_plot(trajectories, trajectory_plot, panels=trajectory_panels)


if __name__ == "__main__":
    main()
