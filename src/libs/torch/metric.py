import torch


def dcg_score(y_score: torch.Tensor, y_true: torch.Tensor, k=10) -> torch.Tensor:
    y_true = y_true.float()
    y_score = y_score.float()

    order = torch.argsort(y_score).flip([0])
    y_true = torch.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = torch.log2(torch.arange(len(y_true), device=y_true.device).float() + 2)

    return torch.sum(gains / discounts)


def ndcg_score(y_score: torch.Tensor, y_true: torch.Tensor, k=10) -> torch.Tensor:
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_score, y_true, k)
    return actual / best
