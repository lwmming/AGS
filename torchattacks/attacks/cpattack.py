import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from ..attack import Attack

mid_outputs = []
class CpAttack(Attack):
    r"""
    cross-paradigm attack
    """
    def __init__(self, model, pretrained, eps=8/255, alpha=1/255, steps=8, momentum=0.9, targeted=False):
        super(CpAttack, self).__init__("CpAttack", model)
        self.eps = eps
        self.steps = steps
        self.momentum = momentum
        self.tar = targeted
        if self.tar:
            self.alpha = -alpha
        else:
            self.alpha = alpha
        self.use_pretrained = pretrained

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        last_grad = torch.zeros_like(images).detach().cuda()
        adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-self.eps, self.eps)
        # feature_layers = list(self.model[1]._modules.keys())
        #feature_layers = list(self.model[1]._modules.keys())[7:8]  # res101
        if not self.use_pretrained:
            feature_layers = ['5']  # res101
        else:
            feature_layers = ['layer2']  # res101

        #mid_outputs = []
        global mid_outputs
        def get_mid_output(m, i, o):
            global mid_outputs
            mid_outputs.append(o)

        hs = []
        for layer_name in feature_layers:
            #print(layer_name)
            if not self.use_pretrained:
                hs.append(self.model.f._modules.get(layer_name).register_forward_hook(get_mid_output))
            else:
                hs.append(self.model[1]._modules.get(layer_name).register_forward_hook(get_mid_output))

        out = self.model(images)

        # import ipdb;ipdb.set_trace()
        mid_originals = []
        for mid_output in mid_outputs:
            mid_original = torch.zeros(mid_output.size()).cuda()
            mid_originals.append(mid_original.copy_(mid_output))
        mid_outputs = []

        for i in range(self.steps):
            adv_images.requires_grad = True
            #tmpp = input_diversity2(adv_images, 0.9, 299, 330)
            outputs = self.model(adv_images)

            mid_originals_ = []
            for mid_original in mid_originals:
                mid_originals_.append(mid_original.detach())
            # import ipdb;ipdb.set_trace()
            n_img = mid_originals_[0].shape[0]
            loss_mid = 1 - F.cosine_similarity(mid_originals_[0].reshape(n_img, -1), mid_outputs[0].reshape(n_img, -1)).mean()

            # print('mid:', loss_mid.item())

            cost = loss_mid.cuda()
            # print(cost)
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            #print(grad.mean())
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + last_grad*self.momentum
            last_grad = grad
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            mid_outputs = []
        print(cost)
        for h in hs:
            h.remove()

        return adv_images
