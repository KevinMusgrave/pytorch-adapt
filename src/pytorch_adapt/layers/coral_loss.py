import torch


def covariance(x):
    batch_size = x.shape[0]
    mm1 = torch.mm(x.t(), x)
    cols_summed = torch.sum(x, dim=0)
    mm2 = torch.mm(cols_summed.unsqueeze(1), cols_summed.unsqueeze(0))
    return (1.0 / (batch_size - 1)) * (mm1 - (1.0 / batch_size) * mm2)


class CORALLoss(torch.nn.Module):
    def forward(self, x, y):
        embedding_size = x.shape[1]
        cx = covariance(x)
        cy = covariance(y)
        squared_fro_norm = torch.linalg.norm(cx - cy, ord="fro") ** 2
        return squared_fro_norm / (4 * (embedding_size ** 2))
