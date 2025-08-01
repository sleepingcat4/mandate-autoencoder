import torch.nn.functional as F
def sparse_loss(y_true, y_pred, encoder, sparsity_level=0.05, lambda_sparse=1e-3):
    mse_loss =  F.mse_loss(y_pred, y_true)
    hidden_output = encoder