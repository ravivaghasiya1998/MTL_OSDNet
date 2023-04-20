import torch
import torch.nn.functional as F
import numpy as np


def InvHuberLoss(logits,target,ignore_index=0):
    #logits=F.relu(logits) # depth predictions must be >=0
    #logits=F.softmax(logits)
    #input=input.cuda()
    target=target/255
    logits =logits.squeeze()
    #logits=logits.reshape(target.shape)
    diff=logits-target
    mask=target !=ignore_index
    
    error=torch.abs(diff*mask.float())
    c=0.2*torch.max(error)
    error2=(diff**2+c**2)/(2.0*c)
    mask_error=error <= c
    mask_error2 = error > c
    loss=torch.mean(error*mask_error.float() + error2*mask_error2.float())
    return loss

def cross_entropy(logits, targets):
    """
    cross entropy loss with one/all hot encoded targets -> logits.size()=targets.size()
    :param logits: torch tensor with logits obtained from network forward pass
    :param targets: torch tensor one/all hot encoded
    :return: computed loss
    """
    #targets = targets.to('cuda')
    #targets = targets.cuda(device)
    neg_log_like = - 1.0 * F.log_softmax(logits, 0)
    #neg_log_like = - 1.0 * F.log_softmax(logits, 1)#dim=1 means sum of probabilites along rows is 1
    L = torch.mul(targets.float(), neg_log_like)
    L = L.mean()
    return L


def encode_target(target, pareto_alpha, num_classes, ignore_train_ind, ood_ind=254):
    """
    encode target tensor with all hot encoding for OoD samples
    :param target: torch tensor
    :param pareto_alpha: OoD loss weight
    :param num_classes: number of classes in original task
    :param ignore_train_ind: void class in original task
    :param ood_ind: class label corresponding to OoD class
    :return: one/all hot encoded torch tensor
    """

    npy = target.cpu().numpy()
    npz = npy.copy()
    npy[np.isin(npy, ood_ind)] = num_classes
    npy[np.isin(npy, ignore_train_ind)] = num_classes + 1
    # npy[np.isin(npy, 100)] = num_classes + 1
    enc = np.eye(num_classes + 2)[npy][..., :-2]  # one hot encoding with last 2 axis cutoff
    enc[(npy == num_classes)] = np.full(num_classes, pareto_alpha / num_classes)  # set all hot encoded vector
    enc[(enc == 1)] = 1 - pareto_alpha  # convex combination between in and out distribution samples
    enc[np.isin(npz, ignore_train_ind)] = np.zeros(num_classes)
    # enc[np.isin(npz, 100)] = np.zeros(num_classes)
    enc = torch.from_numpy(enc)
    enc = enc.permute(0, 3, 1, 2).contiguous()
    return enc
