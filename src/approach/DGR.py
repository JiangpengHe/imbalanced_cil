import torch
import warnings
from copy import deepcopy
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
import numpy as np
import torch.nn as nn

# Distribution-aware-knowledge distillation ---------------------------------------
def _DAKD(pred, soft, cls_prior, T):

    imbalance_rat = measure_imbalance(cls_prior)
    cls_prior = cls_prior / np.sum(cls_prior)
    cls_prior = cls_prior / np.linalg.norm(cls_prior)
    cls_prior = torch.FloatTensor(cls_prior).cuda()
    pred_new = torch.mul(pred, cls_prior) + torch.mul(soft, 1 - cls_prior)
    pred_og = torch.log_softmax(pred / T, dim=1)
    pred_new = torch.log_softmax(pred_new / T, dim=1)
    soft_new = torch.softmax(soft / T, dim=1)
    p_final = imbalance_rat*pred_og + (1 - imbalance_rat)*pred_new
    return -1 * torch.mul(soft_new, p_final).sum() / pred_new.shape[0]


def measure_imbalance(removed_sample):
    removed_sample = removed_sample + 0.001
    n = np.sum(removed_sample)
    
    H = -sum([(temp_sample/n) * np.log((temp_sample/n)) for temp_sample in removed_sample])
    return H/np.log(len(removed_sample))


class Appr(Inc_Learning_Appr):

    def __init__(self, model, device, nepochs=90, lr=0.1, lr_min=1e-6, lr_factor=10, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1.0, T=2, lr_finetuning_factor=0.1,
                 nepochs_finetuning=0, noise_grad=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.lr_finetuning_factor = lr_finetuning_factor
        self.nepochs_finetuning = nepochs_finetuning
        self.noise_grad = noise_grad

        self._train_epoch = 0
        self._finetuning_balanced = None

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: the method is expected to use exemplars. Check the original paper.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def _train_unbalanced(self, t, trn_loader, val_loader):
        """Unbalanced training"""
        self._finetuning_balanced = False
        self._train_epoch = 0
        loader = self._get_train_loader(trn_loader, False)
        super().train_loop(t, loader, val_loader)
        return loader

    def _get_train_loader(self, trn_loader, balanced=False):
        """Modify loader to be balanced or unbalanced"""
        exemplars_ds = self.exemplars_dataset
        trn_dataset = trn_loader.dataset
        if balanced:
            indices = torch.randperm(len(trn_dataset))
            trn_dataset = torch.utils.data.Subset(trn_dataset, indices[:len(exemplars_ds)])
        ds = exemplars_ds + trn_dataset
        return DataLoader(ds, batch_size=trn_loader.batch_size,
                              shuffle=True,
                              num_workers=trn_loader.num_workers,
                              pin_memory=trn_loader.pin_memory)

    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        """Add noise to the gradients"""
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(p.grad.shape, device=p.grad.device) * variance)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if t == 0:  # First task is simple training
            super().train_loop(t, trn_loader, val_loader)
            loader = trn_loader
        else:
            loader = self._train_unbalanced(t, trn_loader, val_loader)
        # After task training： update exemplars
        self.exemplars_dataset.collect_exemplars(self.model, loader, val_loader.dataset.transform)
        if t==0:
            torch.save(self.model.state_dict(),'modeltask0gw.pt')

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images = images.to(self.device)
            # Forward old model
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images)
            outputs = self.model(images)
            self.per_cls_weights = np.array(self.num_per_cls[self.taskidx]) / np.sum(np.array(self.num_per_cls[self.taskidx]))
            loss_cn = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            loss_kd = 0.
            if t > 0:
                loss_kd = self.criterionkd(t, outputs, targets.to(self.device), outputs_old)
            loss = loss_cn + loss_kd
            if t > 0:
                grad_kd = []
                bias_kd = []
                grad_norm_kd = []
                for j in range (len(self.model.heads) - 1):
                    temp_norm = torch.autograd.grad(loss_kd, self.model.heads[j].weight, retain_graph=True)[0]
                    grad_norm_kd.extend(torch.norm(temp_norm, dim =1).data.cpu().numpy())
                    grad_kd.append(temp_norm)
                    bias_kd.append(torch.autograd.grad(loss_kd, self.model.heads[j].bias, retain_graph=True)[0])
                grad_cn = []
                bias_cn = []
                grad_norm_cn = []
                for ele in self.model.heads:
                    temp_norm = torch.autograd.grad(loss_cn, ele.weight, retain_graph=True)[0] 
                    grad_norm_cn.extend(torch.norm(temp_norm, dim =1).data.cpu().numpy())
                    grad_cn.append(temp_norm)
                    bias_cn.append(torch.autograd.grad(loss_cn, ele.bias, retain_graph=True)[0] )
                self.gradall += np.array(grad_norm_cn)
                idx  = 0
                self.grad_weight = []
                task_grad = []
                # sep grad reweight ----------------------
                old_temp = np.array(self.gradall[:len(self.num_per_cls[t-1])])
                new_temp = np.array(self.gradall[len(self.num_per_cls[t-1]):])
                old_temp_weight = np.min(old_temp)/old_temp
                new_temp_weight = np.min(new_temp)/new_temp
                self.grad_weight.extend(old_temp_weight.tolist())
                self.grad_weight.extend(new_temp_weight.tolist())
                self.grad_weight = np.array(self.grad_weight)
                mean_old = np.mean(old_temp)
                mean_new = np.mean(new_temp)
                rs_old = min(1., (mean_new / mean_old))
                rs_new = min(1., (mean_new / mean_old) * self.gamma)
                self.grad_weight[:len(self.num_per_cls[t-1])] = self.grad_weight[:len(self.num_per_cls[t-1])] *  rs_old
                self.grad_weight[len(self.num_per_cls[t-1]):] = self.grad_weight[len(self.num_per_cls[t-1]):] * rs_new
                # to balance the magnitude between knowledge distillation ---------------------
                grad_old = []
                grad_new = []
                grad_both = []
                for k in range (len(self.num_per_cls[t-1])):
                    grad_both.append(self.grad_weight[k] * grad_norm_cn[k])
                for k in range (len(self.num_per_cls[t-1]), len(self.num_per_cls[t])):
                    grad_both.append(self.grad_weight[k] * grad_norm_cn[k])
                ratio_kd = np.linalg.norm(grad_both) / np.linalg.norm(grad_norm_kd)
                # backprop-----------
                self.optimizer.zero_grad()
                loss.backward()
                idx_grad = 0
                for i in range (len(self.model.heads)):
                    for j in range(self.model.heads[i].weight.shape[0]):
                        if i ==  len(self.model.heads)-1:
                            grad_temp_sum =  self.grad_weight[j + idx_grad] * grad_cn[i][j,:]
                        else:
                            grad_temp_sum = self.grad_weight[j + idx_grad] * grad_cn[i][j,:] + ratio_kd * grad_kd[i][j,:] 
                        self.model.heads[i].weight.grad[j, :] = grad_temp_sum

                    for j in range(len(self.model.heads[i].bias)):

                        if i ==  len(self.model.heads)-1:
                            grad_temp_sum =  self.grad_weight[j + idx_grad] * bias_cn[i][j]
                        else:
                            grad_temp_sum = self.grad_weight[j + idx_grad] * bias_cn[i][j] + ratio_kd * bias_kd[i][j]
                        self.model.heads[i].bias.grad[j] = grad_temp_sum
                    idx_grad += self.model.heads[i].weight.shape[0]
            else:
                self.optimizer.zero_grad()
                loss.backward()

                grad_norm = []
                for ele in self.model.heads:
                    grad_norm.extend(torch.norm(ele.weight.grad, dim =1).data.cpu().numpy())
                # accumulated gradients 
                self.gradall += np.array(grad_norm)
                # class-balance ratio
                self.grad_weight = np.min(self.gradall) / np.array(self.gradall)
                idx_grad = 0
                for i in range(len(self.model.heads)):
                    for j in range(self.model.heads[i].weight.shape[0]):
                        self.model.heads[i].weight.grad[j, :] = self.model.heads[i].weight.grad[j, :] * self.grad_weight[j + idx_grad] 
                    for j in range(len(self.model.heads[i].bias)):
                        self.model.heads[i].bias.grad[j] = self.model.heads[i].bias.grad[j] * self.grad_weight[j + idx_grad] 
                    idx_grad += self.model.heads[i].weight.shape[0]
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            if self.noise_grad:
                self._noise_grad(self.model.parameters(), self._train_epoch)
            self.optimizer.step()
        self._train_epoch += 1
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        self.prior_regularization = self.num_per_cls[self.taskidx] / np.sum(self.num_per_cls[self.taskidx])
        self.prior_regularization = torch.FloatTensor(self.prior_regularization).cuda()
        self.prior_regularization = torch.log(self.prior_regularization)
        outputs_loss = torch.cat(outputs, dim=1)
        outputs_loss += self.prior_regularization
        loss = torch.nn.functional.cross_entropy(outputs_loss, targets)
        return loss
    def criterionkd(self, t, outputs, targets, outputs_old=None):
        if t > 0 and outputs_old:
            loss = self.kd_ratio * 2 * 2 * _DAKD(torch.cat(outputs[:-1], dim=1), torch.cat(outputs_old, dim=1), self.kdprior, 2)
            return loss
