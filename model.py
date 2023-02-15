import torch
import torch.nn as nn
import numpy as np
import math


def num_params(net):
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    num_params = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    return num_params


class RSA(nn.Module):
    def __init__(self, in_channels, out_channels, r=2, s=2, hidden_channels=32, max_L=100):
        super().__init__()

        n_heads = r + 2 * s
        self.r = r
        self.s = s
        self.n_heads = n_heads
        self.hidden_channels = hidden_channels
        self.max_L = max_L

        self.mu = nn.Parameter(torch.rand(1) * 6. - 3.)

        flag = torch.randint(-1, 1, (r,))
        flag = torch.where(flag < 0, flag, 1).float()
        self.lam = nn.Parameter(((torch.rand(r)+ 1.) * flag).unsqueeze(1).unsqueeze(1))

        self.gamma_c1 = nn.Parameter((torch.tanh((torch.rand(s)+ 1.))).unsqueeze(1).unsqueeze(1))
        self.gamma_c2 = nn.Parameter((torch.tanh((torch.rand(s)+ 1.))).unsqueeze(1).unsqueeze(1))
        self.theta_c1 = nn.Parameter((torch.tensor([math.pi / 4] * s)).unsqueeze(1).unsqueeze(1))
        self.theta_c2 = nn.Parameter((torch.tensor([math.pi / 4] * s)).unsqueeze(1).unsqueeze(1))

        self.proj_q = nn.Linear(in_channels, hidden_channels * n_heads)
        self.proj_k = nn.Linear(in_channels, hidden_channels * n_heads)
        self.proj_v = nn.Linear(in_channels, hidden_channels * n_heads)
        self.proj_final = nn.Linear(hidden_channels * n_heads, out_channels)

        self.sigmoid = nn.Sigmoid()

    def build_real_positional_embedding(t, r, L, lam):
        L = L.repeat(r, 1, 1)
        lam_full = lam.repeat(1, t, t)
        P = lam_full.pow(L).tril(diagonal=-1)
        return P

    def build_complex_positional_embedding(t, s, L, gamma, theta, func):
        L = L.repeat(s, 1, 1)
        gamma_full = gamma.repeat(1, t, t)
        P = gamma_full.pow(L).tril(diagonal=-1)
        P = P * func(theta * L)
        return P

    def build_positional_embedding(self, t, device):
        L = torch.tril(torch.ones(t, t).to(device), diagonal=-1)
        L = L.cumsum(0)
        L = torch.where(L > self.max_L, self.max_L, L)
        P_R = RSA.build_real_positional_embedding(t, self.r, L, self.lam)
        P_C1 = RSA.build_complex_positional_embedding(t, self.s, L, self.gamma_c1, self.theta_c1, torch.sin)
        P_C2 = RSA.build_complex_positional_embedding(t, self.s, L, self.gamma_c2, self.theta_c2, torch.cos)

        P_R = P_R
        P_C1 = P_C1
        P_C2 = P_C2
        P = torch.cat([P_R, P_C1, P_C2], dim=0)
        return P
        
    def forward(self, x: torch.Tensor):
        b, t, d = x.shape
        P = self.build_positional_embedding(t, x.device).repeat(b, 1, 1, 1)

        Q = self.proj_q(x)
        K = self.proj_k(x)
        V = self.proj_v(x)

        Q = Q.view(b, t, self.n_heads, self.hidden_channels).transpose(1, 2)
        K = K.view(b, t, self.n_heads, self.hidden_channels).transpose(1, 2)
        V = V.view(b, t, self.n_heads, self.hidden_channels).transpose(1, 2)

        att_score = torch.softmax(torch.matmul(Q, K.transpose(2, 3)), dim=3)

        weight = self.sigmoid(self.mu)

        x = torch.matmul((1. - weight) * att_score + weight * P, V)
        x = x.transpose(1, 2).contiguous().view(b, t, -1)
        x = self.proj_final(x)

        return x



if __name__ == '__main__':
    gru = nn.GRU(640, 256, num_layers=2, batch_first=True)
    rsa = RSA(640, 256, r=4, s=2, hidden_channels=64)
    x = torch.rand(3, 512, 640)
    y1 = rsa(x)
    y2 = gru(x)[0]
    print(y1.shape, y2.shape)
    print(num_params(gru))
    print(num_params(rsa))
    print(y1.isnan().sum(), y1.isinf().sum())
