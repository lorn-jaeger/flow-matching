import datetime
import math
import pathlib
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class MNISTFlowDataset:
    def __init__(self, root="./data", train=True, download=True):
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )

    def make_loader(self, batch_size=256, num_workers=2):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )


class ConditionalMNISTVectorFieldMLP(nn.Module):
    def __init__(self, hidden_size=1024, embedding_dim=32):
        super().__init__()
        self.label_emb = nn.Embedding(10, embedding_dim)
        in_features = (28 * 28) + 1 + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 28 * 28),
        )

    def forward(self, x_t, t, labels):
        batch = x_t.shape[0]
        flat = x_t.view(batch, -1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        emb = self.label_emb(labels)
        inputs = torch.cat([flat, t, emb], dim=-1)
        out = self.net(inputs)
        return out.view(batch, 1, 28, 28)


def flow_matching_loss(model, data_batch, labels):
    noise = torch.randn_like(data_batch)
    t = torch.rand(data_batch.shape[0], 1, device=data_batch.device)
    t_map = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_map) * noise + t_map * data_batch
    target_velocity = data_batch - noise
    predicted_velocity = model(x_t, t, labels)
    return torch.mean((predicted_velocity - target_velocity) ** 2)


@torch.no_grad()
def sample_digits(model, num_samples=64, label=None, integration_steps=80, device="cpu"):
    model.eval()
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    if label is None:
        labels = torch.randint(0, 10, (num_samples,), device=device)
    else:
        labels = torch.full((num_samples,), label, device=device, dtype=torch.long)

    dt = 1.0 / integration_steps
    for step in range(integration_steps):
        t_scalar = torch.full((num_samples, 1), step / integration_steps, device=device)
        velocity = model(x, t_scalar, labels)
        x = x + dt * velocity

    imgs = x.clamp(0.0, 1.0)
    return imgs.cpu(), labels.cpu()


def save_image_grid(tensor, out_path, nrow=8, normalize=True):
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(tensor, out, nrow=nrow, normalize=normalize)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def archive_file(src_path, archive_dir="archive"):
    if archive_dir is None:
        return
    archive_dir = pathlib.Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    src = pathlib.Path(src_path)
    candidate = archive_dir / f"{timestamp}_{src.name}"
    counter = 1
    while candidate.exists():
        candidate = archive_dir / f"{timestamp}_{counter}_{src.name}"
        counter += 1
    shutil.copy2(src, candidate)
    print(f"Archived {src.name} to {candidate}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    dataset = MNISTFlowDataset("./data", train=True, download=True)
    loader = dataset.make_loader(batch_size=256)
    loader_iter = infinite_loader(loader)

    model = ConditionalMNISTVectorFieldMLP(
        hidden_size=1024,
        embedding_dim=32,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_steps = 5000
    log_every = 250

    for step in range(1, train_steps + 1):
        batch, labels = next(loader_iter)
        batch = batch.to(device)
        labels = labels.to(device)

        loss = flow_matching_loss(model, batch, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            print(f"step {step:05d} | loss {loss.item():.4f}")

    random_imgs, _ = sample_digits(
        model,
        num_samples=64,
        integration_steps=80,
        device=device,
    )
    save_image_grid(random_imgs, "mnist_flow_samples.png", nrow=int(math.sqrt(64)))
    archive_file("mnist_flow_samples.png")

    class_grids = []
    for digit in range(10):
        imgs, _ = sample_digits(
            model,
            num_samples=32,
            label=digit,
            integration_steps=80,
            device=device,
        )
        class_grids.append(imgs)

    per_class = torch.cat(class_grids, dim=0)
    save_image_grid(per_class, "mnist_flow_samples_per_class.png", nrow=32)
    archive_file("mnist_flow_samples_per_class.png")


if __name__ == "__main__":
    main()
