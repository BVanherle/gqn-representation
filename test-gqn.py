import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from gqn import GenerativeQueryNetwork
import random
from data.shepardmetzler import ShepardMetzler
from torch.utils.data import DataLoader
from data.utils import deterministic_partition


def query_context_plot(images, indices):
    # Visualise context and query images
    f, axarr = plt.subplots(1, 15, figsize=(20, 7))
    for i, ax in enumerate(axarr.flat):
        # Move channel dimension to end
        ax.imshow(images[scene_id][i].permute(1, 2, 0))

        if i == indices[-1]:
            ax.set_title("Query", color="magenta")
        elif i in indices[:-1]:
            ax.set_title("Context", color="green")
        else:
            ax.set_title("Unused", color="grey")

        ax.axis("off")


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ShepardMetzler("I:/Datasets/GQN/shepard_metzler_5_parts", train=False, use_pos_enc=True)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
state_dict = torch.load("./trained_models/shepardmetzler_200.pt")

model_settings = dict(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)
model = GenerativeQueryNetwork(**model_settings)

# Load trained parameters, un-dataparallel if needed
if True in ["module" in m for m in list(state_dict.keys())]:
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.module
else:
    model.load_state_dict(state_dict["model"])

x, v = next(iter(loader))
x_, v_ = x.squeeze(0), v.squeeze(0)

# Sample a set of views
n_context = 7 + 1

for scene_id in range(x_.shape[0]):
    indices = random.sample([i for i in range(v_.size(1))], n_context)
    x_c, v_c, x_q, v_q = deterministic_partition(x, v, indices)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 7))

    x_mu, r, kl = model(x_c[scene_id].unsqueeze(0),
                        v_c[scene_id].unsqueeze(0),
                        x_q[scene_id].unsqueeze(0),
                        v_q[scene_id].unsqueeze(0))

    x_mu = x_mu.squeeze(0)
    r = r.squeeze(0)

    ax1.imshow(x_q[scene_id].data.permute(1, 2, 0))
    ax1.set_title("Query image")
    ax1.axis("off")

    ax2.imshow(x_mu.data.permute(1, 2, 0))
    ax2.set_title("Reconstruction")
    ax2.axis("off")

    ax3.imshow(r.data.view(16, 16))
    ax3.set_title("Representation")
    ax3.axis("off")

    plt.show()