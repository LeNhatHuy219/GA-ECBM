"""GA-ECBM (gated additive ECBM) inside the official ECBM code (xmed-lab/ECBM, commit d38c43a).

Kept from the official code: the ResNet-101 backbone, the image-label energy E_class and the image-concept energy
E_concept of EBM_GL.
Replaced: E_global(c, y) = -(b_y + sum_k theta_{y,k} * c_k), an additive score table instead of the MLP over the
whole concept vector. Because the energy is additive, p(y | x, verified concepts) and p(c_k | x, ...) are exact and
closed-form, with no gradient-based optimization.
When k concept groups are verified, the image branch is multiplied by the gate w(k) = exp(-sum_{j<=k} softplus(a_j)),
w(0) = 1.

This file is the code that ran on Kaggle with comments translated to English; the notebooks write the original
(Vietnamese-commented) version into the cloned official repository. tests/check_cub_modules.py verifies that both
versions have the same syntax tree apart from docstrings.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.cub import CONCEPT_GROUP_MAP
from networks.EBM import ECBM

GROUPS = [list(indices) for indices in CONCEPT_GROUP_MAP.values()]   # 28 groups, as in the official group intervention
GATE_INIT = float(os.environ.get("GA_GATE_INIT", "-3.1"))             # softplus(-3.1) ~ 0.044 -> w(28) ~ 0.29


class GAECBM(ECBM):
    def __init__(self, args, emb_size, cpt_size, hid_size, n_classes):
        super().__init__(args, emb_size=emb_size, cpt_size=cpt_size, hid_size=hid_size, n_classes=n_classes)
        head = self.energy_model
        del head.concept_proj, head.fc_c, head.classifier_cy        # official E_global MLP: no longer used
        self.theta = nn.Parameter(torch.zeros(n_classes, cpt_size))  # theta_{y,k}: score of concept k = 1 for class y
        self.label_bias = nn.Parameter(torch.zeros(n_classes))       # b_y
        self.gate_steps = nn.Parameter(torch.full((len(GROUPS),), GATE_INIT))
        group_of = torch.zeros(cpt_size, dtype=torch.long)
        for g, indices in enumerate(GROUPS):
            group_of[indices] = g
        self.register_buffer("group_of", group_of, persistent=False)

    def energies(self, images):
        """Official E_class [B, C] and E_concept [B, K, 2]; E_concept[..., s] is the energy of state s."""
        emb = self.backbone(images)
        e_class, _, e_concept = self.energy_model(emb, None, True, use_cy=False)
        return e_class, e_concept

    def gate(self, verified_groups):
        w = torch.exp(-torch.cat([self.gate_steps.new_zeros(1), torch.cumsum(F.softplus(self.gate_steps), 0)]))
        return w[verified_groups]

    def global_score(self, concepts):
        """-E_global(c, y) for every y: [B, C]."""
        return self.label_bias + concepts.float() @ self.theta.t()

    def posterior(self, e_class, e_concept, mask=None, values=None, class_weight=None):
        """log p(y | x, c_S) [B, C] and p(c_k = 1 | x, c_S) [B, K]; computed in FP32."""
        e_class, e_concept = e_class.float(), e_concept.float()
        t1 = -e_concept[..., 1][:, None, :] + self.theta[None]          # [B, C, K]
        t0 = (-e_concept[..., 0])[:, None, :].expand_as(t1)
        per_concept = torch.logaddexp(t0, t1)
        p1_given_y = torch.sigmoid(t1 - t0)
        if mask is not None:
            observed = torch.where(values[:, None, :].bool(), t1, t0)
            per_concept = torch.where(mask[:, None, :], observed, per_concept)
            p1_given_y = torch.where(mask[:, None, :], values[:, None, :].float(), p1_given_y)
        weight = 1.0 if class_weight is None else class_weight.float()[:, None]
        log_p_y = (-weight * e_class + self.label_bias[None] + per_concept.sum(-1)).log_softmax(-1)
        return log_p_y, (log_p_y.exp()[:, :, None] * p1_given_y).sum(1)


def ga_losses(model, loss_label, loss_concept, images, labels, concepts, weights=(3, 1, 1)):
    """GA-ECBM loss for one batch.

    Returns (training loss, validation loss as in the official code, components, (e_class, e_concept, clean concepts)).
    The official 3 : 1 : 1 loss calls the official loss functions and cy_augment. In the official forward pass,
    cy_augment modifies the batch concept tensor in place, so the official L_xc is also computed on perturbed concepts;
    this behaviour is kept for the official loss terms. The three added losses use clean concepts (a copy taken before
    the perturbation).
    """
    clean = concepts.clone()
    head = model.energy_model
    e_class, e_concept = model.energies(images)
    perturbed = head.cy_augment(c_gt=concepts.unsqueeze(-1), permute_ratio=head.cy_concept_perturb_prob,
                                permute_prob=head.cy_sample_perturb_prob).squeeze(-1)
    loss_xy = loss_label(e_class, labels)
    loss_cy = loss_label(-model.global_score(perturbed), labels)       # EBMLoss_label takes energies
    loss_xc = loss_concept(e_concept, concepts)
    w_xy, w_xc, w_cy = weights
    base = w_xy * loss_xy + w_xc * loss_xc + w_cy * loss_cy

    device_type = "cuda" if images.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        count, n_groups = len(labels), len(GROUPS)
        log_p_y, p_c = model.posterior(e_class, e_concept)
        joint = F.nll_loss(log_p_y, labels) + F.binary_cross_entropy(p_c.clamp(1e-6, 1 - 1e-6), clean.float())
        groups = torch.rand(count, n_groups, device=images.device) < torch.rand(count, 1, device=images.device)
        drop = torch.rand(count, device=images.device) < 0.5                  # half of the cases switch the image branch off
        k = groups.sum(1)
        w_k = torch.where(drop, torch.zeros_like(k, dtype=torch.float), model.gate(k))
        log_int, _ = model.posterior(e_class, e_concept, groups[:, model.group_of], clean, w_k)
        loss_int = F.nll_loss(log_int, labels, reduction="none")
        extra = torch.multinomial((~groups).float() + 1e-9, 1).squeeze(1)      # verify one more group
        groups_more = groups.clone()
        groups_more[torch.arange(count, device=images.device), extra] = True
        k_more = groups_more.sum(1)
        w_more = torch.where(drop, torch.zeros_like(k_more, dtype=torch.float), model.gate(k_more))
        log_more, _ = model.posterior(e_class, e_concept, groups_more[:, model.group_of], clean, w_more)
        mono = F.relu(F.nll_loss(log_more, labels, reduction="none") - loss_int).mean()
    total = base + joint + loss_int.mean() + mono
    val_like = loss_xc + loss_xy + loss_cy                                   # as in the official validation_step
    parts = {"xy": loss_xy, "xc": loss_xc, "cy": loss_cy, "joint": joint, "int": loss_int.mean(), "mono": mono}
    return total, val_like, parts, (e_class, e_concept, clean, log_p_y, p_c)
