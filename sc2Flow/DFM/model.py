import torch
from torch import nn, Tensor
import math
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Model class
class MLP(nn.Module):
    def __init__(
            self, input_dim: int = 128, time_dim: int = 1, hidden_dim=128, length=2):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.time_embedding = nn.Linear(1, time_dim)
        self.token_embedding = torch.nn.Embedding(self.input_dim, hidden_dim)

        self.main = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim * length + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, self.input_dim * length),
        )

    def forward(self, x, t):
        t = self.time_embedding(t.unsqueeze(-1))
        x = self.token_embedding(x)

        B, N, d = x.shape
        x = x.reshape(B, N * d)

        h = torch.cat([x, t], dim=1)
        h = self.main(h)

        h = h.reshape(B, N, self.input_dim)

        return h

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, dtype=torch.float32):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(dim, dim, dtype=dtype),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(dim, dim, dtype=dtype),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block1(x)
        out = self.block2(x)
        return out + identity


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x

# Model class
class MLP(nn.Module):
    def __init__(
            self, input_dim: int = 128,
            time_dim: int = 1,
            embedding_dim: int = 64,
            hidden_dim=128,
            length=2,
            n_layers=4,
            dropout=0.,
            condition_num=0,
            ):
        super().__init__()

        self.condition_num = condition_num

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.time_embedding = nn.Linear(1, embedding_dim)
        self.token_embedding = torch.nn.Embedding(self.input_dim, embedding_dim)

        self.to_vector = nn.Linear(embedding_dim, 1)
        self.input_linear = nn.Linear(length + 1 + condition_num, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, self.input_dim * length)

        self.main = nn.ModuleList([])
        self.main.append(Swish())
        for i in range(n_layers):
            self.main.append(ResidualBlock(dim=hidden_dim, dropout=dropout))
            self.main.append(Swish())

        #self.t_embedder = TimestepEmbedder(min(128, 256), mid_size=1024)
        self.temp_c_linear = nn.Linear(768, embedding_dim)
        self.dose_embdding = nn.Parameter(torch.randn(1, embedding_dim)) #

    def forward(self, x, t, **labels):
        #t = self.t_embedder(t)
        t = self.time_embedding(t.unsqueeze(-1)).unsqueeze(1)
        x = self.token_embedding(x)

        B, N, d = x.shape

        if labels != {}:
            assert self.condition_num != 0
            cond = labels['c_rep']
            cond = self.temp_c_linear(cond).unsqueeze(1)

            dose = labels['dose']
            #dose = torch.log(dose + 1).unsqueeze(1) @ self.dose_embdding
            dose = dose.unsqueeze(1) @ self.dose_embdding
            dose = dose.unsqueeze(1)
            h = torch.cat([x, t, cond, dose], dim=1)
        else:
            assert self.condition_num == 0
            h = torch.cat([x, t], dim=1)
        h = self.to_vector(h).squeeze(-1)
        h = self.input_linear(h)

        for layer in self.main:
            h = layer(h)

        h = self.output_linear(h)

        h = h.reshape(B, N, self.input_dim)

        return h

def pack1(net,x,t,**pack):
    return pack2(net=net,x=x, t=t, **pack)

def pack2(net,x,t,**pack):
    return net(x=x, t=t, **pack)



if __name__ == "__main__":
    vocab_size = 2
    time_dim =1
    hidden_dim = 128
    dims = 2000
    net = MLP(input_dim=vocab_size, time_dim=1, hidden_dim=hidden_dim, length=dims,condition_num=2)

    bsz = 3
    x = torch.randint(0, 1, (bsz, dims))
    c = torch.randn(bsz, 768)
    d = torch.rand(bsz)
    t = torch.rand(bsz)




    ans = net(x=x,t=t,c_rep=c,dose=d)

    print(ans.shape)