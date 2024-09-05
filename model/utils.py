
import torch

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    all_size = (param_size + buffer_size) / 1024 / 1024
    return param_sum, buffer_sum, all_size


def compute_accu_matrix(a, b):
    TP = torch.sum((a == True) & (b == True), 0)
    # True negative (TN): we predicted a negative result and it was negative
    TN = torch.sum((a == False) & (b == False), 0)

    # False positive (FP): we predicted a positive result and it was negative
    FP = torch.sum((a == True) & (b == False), 0)

    # False negative (FN): we predicted a negative result and it was positive
    FN = torch.sum((a == False) & (b == True), 0)

    eps = 1e-7
    # True positive rate (TPR)
    TPR = TP / (TP + FN + eps)

    # False positive rate (FPR)
    FPR = FP / (FP + TN + eps)

    # True Negative Rate (TNR)
    TNR = TN / (TN + FP + eps)

    # False Negative Rate (FNR)
    FNR = FN / (FN + TP + eps)

    ACC = (TP + TN) / (TP + TN + FP + FN)

    return {'acc': ACC, 'a_tpr': TPR, 'a_fpr': FPR, 'a_tnr': TNR, 'a_fnr': FNR, }
