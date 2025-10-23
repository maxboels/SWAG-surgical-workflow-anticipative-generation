import torch

def accuracy_n_pred(output, target, n_pred=1, ignore_index=7):
    """Compute the accuracies for every predictions.
    Accuracies:
        - per prediction index (element-wise accuracy)
        - up-to a prediction index (cumulative accuracy)
    Args:
        output (torch.Tensor): The model output of shape (B, N, C).
        target (torch.Tensor): The target of shape (B, N).
        n_pred (int): The number of predictions.
        ignore_index (int): The index to ignore. Default: 7 (no next action class).
    Returns:
        dict: A dictionary containing the element-wise accuracy and the cumulative accuracy.
    """
    # Element-wise accuracy
    pred = output.argmax(dim=-1)
    correct = pred.eq(target)
    mask = target.ne(ignore_index)
    correct = correct * mask
    acc = []
    for i in range(n_pred):
        acc.append(correct[:, i].sum().float() / mask[:, i].sum().float())

    # Cumulative accuracy
    correct_cumulative = []
    for i in range(n_pred):
        correct_cumulative.append(correct[:, :i + 1].sum().float() / mask[:, :i + 1].sum().float())
    correct_cumulative = torch.stack(correct_cumulative)

    return {"acc": acc, "acc_cumulative": correct_cumulative}