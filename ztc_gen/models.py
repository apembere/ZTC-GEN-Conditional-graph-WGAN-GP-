"""
Conditional WGAN-GP (and conditional VAE) on dense ZTC graph tensors.

Shapes:   X  (B, N, F)   node feature matrix        N = Nmax = 150, F = 19
          A  (B, N, N, R) adjacency, one-hot bonds    R = 4 (single, double, triple, aromatic)
          p  (B, d)       z-scored conditioning properties
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .featurize import FEATURE_BLOCKS


def mlp(sizes, act=nn.LeakyReLU(0.2), norm=None, dropout=0.0, last_act=True):
    layers = []
    for k in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[k], sizes[k + 1]))
        if k < len(sizes) - 2 or last_act:
            if norm == "layer":
                layers.append(nn.LayerNorm(sizes[k + 1]))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(sizes[k + 1]))
            layers.append(act)
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def activate_nodes(x_logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
    """Per-block output activation: softmax for one-hot blocks, sigmoid for binary/scalar."""
    out, s = [], 0
    for _, w, kind in FEATURE_BLOCKS:
        blk = x_logits[..., s:s + w]
        if kind == "onehot":
            blk = F_.gumbel_softmax(blk, tau=1.0, hard=True, dim=-1) if hard else torch.softmax(blk, -1)
        else:
            blk = torch.sigmoid(blk)
        out.append(blk)
        s += w
    return torch.cat(out, -1)


def symmetrize_edges(e_logits: torch.Tensor) -> torch.Tensor:
    """(B,N,N,R) logits -> symmetric bond probabilities in [0,1] with an empty diagonal."""
    e = torch.sigmoid((e_logits + e_logits.transpose(1, 2)) / 2)
    n = e.shape[1]
    mask = 1.0 - torch.eye(n, device=e.device).view(1, n, n, 1)
    return e * mask


class Generator(nn.Module):
    """G(z, p) -> (X, A).

    edge_head = "dense"    : z||p -> dense layers -> flattened N*N*R edge tensor (as in the method text)
    edge_head = "pairwise" : dense layers -> per-node embeddings h_i; edge logits from MLP(h_i*h_j, |h_i-h_j|).
                             ~20x fewer parameters, permutation-equivariant edges; recommended on CPU.
    """

    def __init__(self, nmax, n_feat, n_bond, n_prop, z_dim=128, hidden=(256, 512, 512),
                 edge_head="dense", node_emb=64, dropout=0.0):
        super().__init__()
        self.nmax, self.n_feat, self.n_bond, self.z_dim, self.edge_head = nmax, n_feat, n_bond, z_dim, edge_head
        self.trunk = mlp([z_dim + n_prop, *hidden], act=nn.LeakyReLU(0.2), norm="layer", dropout=dropout)
        h = hidden[-1]
        self.node_out = nn.Linear(h, nmax * n_feat)
        if edge_head == "dense":
            self.edge_out = nn.Linear(h, nmax * nmax * n_bond)
        else:
            self.emb_out = nn.Linear(h, nmax * node_emb)
            self.pair = mlp([2 * node_emb, 128, n_bond], last_act=False)

    def forward(self, z, p, return_logits=False):
        h = self.trunk(torch.cat([z, p], -1))
        B = z.shape[0]
        x_logits = self.node_out(h).view(B, self.nmax, self.n_feat)
        if self.edge_head == "dense":
            e_logits = self.edge_out(h).view(B, self.nmax, self.nmax, self.n_bond)
        else:
            emb = self.emb_out(h).view(B, self.nmax, -1)
            hi, hj = emb.unsqueeze(2), emb.unsqueeze(1)
            pair = torch.cat([(hi * hj).expand(-1, -1, self.nmax, -1),
                              (hi - hj).abs()], -1)
            e_logits = self.pair(pair)
        X, A = activate_nodes(x_logits), symmetrize_edges(e_logits)
        if return_logits:
            return X, A, x_logits, e_logits
        return X, A


class MLPCritic(nn.Module):
    """D(X, A, p): flatten X and A, concatenate with p, fully connected -> scalar Wasserstein score.
    LayerNorm (never BatchNorm) because the gradient penalty is computed per sample."""

    def __init__(self, nmax, n_feat, n_bond, n_prop, hidden=(512, 256, 128), dropout=0.1):
        super().__init__()
        self.adj_proj = nn.Linear(nmax * nmax * n_bond, hidden[0])   # largest layer, kept separate for clarity
        self.node_proj = nn.Linear(nmax * n_feat, hidden[0] // 2)
        self.body = mlp([hidden[0] + hidden[0] // 2 + n_prop, *hidden[1:]], norm="layer", dropout=dropout)
        self.out = nn.Linear(hidden[-1], 1)

    def forward(self, X, A, p):
        B = X.shape[0]
        a = F_.leaky_relu(self.adj_proj(A.reshape(B, -1)), 0.2)
        x = F_.leaky_relu(self.node_proj(X.reshape(B, -1)), 0.2)
        return self.out(self.body(torch.cat([a, x, p], -1))).squeeze(-1)


class GraphConvCritic(nn.Module):
    """Relational graph-convolution critic (MolGAN-style). Optional: `--critic gcn`.
    Permutation-invariant, far fewer parameters than the flattened MLP critic."""

    def __init__(self, nmax, n_feat, n_bond, n_prop, gc_dims=(128, 128, 128), readout=256, hidden=(256, 128)):
        super().__init__()
        self.n_bond = n_bond
        dims = [n_feat, *gc_dims]
        self.rel = nn.ModuleList([nn.Linear(dims[k], dims[k + 1] * n_bond, bias=False) for k in range(len(gc_dims))])
        self.self_lin = nn.ModuleList([nn.Linear(dims[k], dims[k + 1]) for k in range(len(gc_dims))])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in gc_dims])
        self.gate = nn.Linear(gc_dims[-1] + n_feat, readout)
        self.val = nn.Linear(gc_dims[-1] + n_feat, readout)
        self.head = mlp([readout + n_prop, *hidden], norm="layer")
        self.out = nn.Linear(hidden[-1], 1)

    def forward(self, X, A, p):
        B, N, _ = X.shape
        Ar = A.permute(0, 3, 1, 2)                          # (B,R,N,N)
        deg = Ar.sum(-1, keepdim=True).clamp(min=1.0)
        h = X
        for rel, sl, nrm in zip(self.rel, self.self_lin, self.norms):
            m = rel(h).view(B, N, self.n_bond, -1).permute(0, 2, 1, 3)   # (B,R,N,D)
            msg = (torch.matmul(Ar, m) / deg).sum(1)                      # (B,N,D)
            h = torch.tanh(nrm(msg + sl(h)))
        hx = torch.cat([h, X], -1)
        g = (torch.sigmoid(self.gate(hx)) * torch.tanh(self.val(hx))).sum(1)  # gated sum readout
        return self.out(self.head(torch.cat([torch.tanh(g), p], -1))).squeeze(-1)


def build_critic(kind, nmax, n_feat, n_bond, n_prop):
    if kind == "mlp":
        return MLPCritic(nmax, n_feat, n_bond, n_prop)
    if kind == "gcn":
        return GraphConvCritic(nmax, n_feat, n_bond, n_prop)
    raise ValueError(kind)


def gradient_penalty(critic, X_real, A_real, X_fake, A_fake, p, lam=10.0):
    """lambda * E[(||grad_{x_hat} D(x_hat)||_2 - 1)^2], x_hat = eps*x_real + (1-eps)*x_fake."""
    B = X_real.shape[0]
    eps = torch.rand(B, 1, 1, device=X_real.device)
    X_hat = (eps * X_real + (1 - eps) * X_fake).requires_grad_(True)
    A_hat = (eps.unsqueeze(-1) * A_real + (1 - eps.unsqueeze(-1)) * A_fake).requires_grad_(True)
    d_hat = critic(X_hat, A_hat, p)
    gX, gA = torch.autograd.grad(d_hat.sum(), [X_hat, A_hat], create_graph=True)
    gnorm = torch.sqrt(gX.reshape(B, -1).pow(2).sum(1) + gA.reshape(B, -1).pow(2).sum(1) + 1e-12)
    return lam * ((gnorm - 1.0) ** 2).mean(), gnorm.mean().detach()


# ------------------------------------------------------------------------- conditional VAE
class CVAE(nn.Module):
    """Conditional VAE alternative: q(z|X,A,p) encoder + the same Generator as decoder p(X,A|z,p)."""

    def __init__(self, nmax, n_feat, n_bond, n_prop, z_dim=128, edge_head="pairwise"):
        super().__init__()
        self.z_dim = z_dim
        self.enc_graph = GraphConvCritic(nmax, n_feat, n_bond, n_prop, hidden=(512, 256))
        self.enc_graph.out = nn.Identity()
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)
        self.decoder = Generator(nmax, n_feat, n_bond, n_prop, z_dim=z_dim, edge_head=edge_head)

    def encode(self, X, A, p):
        h = self.enc_graph(X, A, p)
        return self.mu(h), self.logvar(h).clamp(-10, 10)

    def forward(self, X, A, p):
        mu, logvar = self.encode(X, A, p)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        Xh, Ah, xl, el = self.decoder(z, p, return_logits=True)
        return Xh, Ah, xl, el, mu, logvar


def cvae_loss(X, A, x_logits, e_logits, mu, logvar, beta=1.0, pos_weight=20.0):
    """Reconstruction (per-block CE/BCE on nodes, weighted BCE on bonds) + beta * KL."""
    loss_x, s = 0.0, 0
    for _, w, kind in FEATURE_BLOCKS:
        xl, xt = x_logits[..., s:s + w], X[..., s:s + w]
        if kind == "onehot":
            loss_x = loss_x + F_.cross_entropy(xl.reshape(-1, w), xt.reshape(-1, w).argmax(-1))
        else:
            loss_x = loss_x + F_.binary_cross_entropy_with_logits(xl, xt)
        s += w
    e_sym = (e_logits + e_logits.transpose(1, 2)) / 2
    pw = torch.tensor(pos_weight, device=A.device)
    loss_a = F_.binary_cross_entropy_with_logits(e_sym, A, pos_weight=pw)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return loss_x + loss_a + beta * kl, dict(rec_x=float(loss_x), rec_a=float(loss_a), kl=float(kl))
