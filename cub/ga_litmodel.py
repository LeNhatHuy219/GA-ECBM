"""GA-ECBM LightningModule: subclasses the official LitModel and keeps its optimizer, scheduler and logging.

main.py changes a single import line to use this class; everything else (data, sampler, ModelCheckpoint, number of
epochs) is unchanged. Comments translated to English; see ga_ecbm.py.
"""
import logging
import os

import torch

from LitModel import LitModel
from ga_ecbm import GATE_INIT, GROUPS, GAECBM, ga_losses


def _accuracy(pred, target):
    return (pred == target).float().mean()


class GALitModel(LitModel):
    def __init__(self, logdir, args):
        super().__init__(logdir, args)
        self.network = GAECBM(args=args, emb_size=args.emb_size, cpt_size=args.cpt_size, hid_size=args.hid_size,
                              n_classes=args.n_classes)
        logging.info("GA-ECBM: gate_init=%s, groups=%d", GATE_INIT, len(GROUPS))   # marker the notebooks use to recognise a GA run

    def _step(self, batch, stage):
        images, labels, concepts = batch
        total, val_like, parts, (e_class, e_concept, clean, log_p_y, p_c) = ga_losses(
            self.network, self.loss_label, self.loss_concept, images, labels, concepts,
            (self.lambda_xy, self.lambda_xc, self.lambda_cy))
        pred_c_direct = torch.argmin(e_concept, 2)
        pred_c_post = (p_c > 0.5).long()
        values = {
            # as in the official code: read the energies directly; concepts compared with the batch tensor (modified by cy_augment)
            f"accuracy_{stage}_y_xy": _accuracy(torch.argmin(e_class, 1), labels),
            f"accuracy_{stage}_c": _accuracy(pred_c_direct, concepts),
            f"accuracy_{stage}_c_overall": (pred_c_direct == concepts).all(1).float().mean(),
            # GA-ECBM: exact inference, compared with the clean concepts
            f"accuracy_{stage}_y_post": _accuracy(log_p_y.argmax(1), labels),
            f"accuracy_{stage}_c_post": _accuracy(pred_c_post, clean),
            f"accuracy_{stage}_c_overall_post": (pred_c_post == clean).all(1).float().mean(),
        }
        for name, value in values.items():
            self.log(name, value, on_epoch=True, on_step=False)
        return total, val_like, parts, values

    def training_step(self, batch, batch_idx):
        total, _, parts, _ = self._step(batch, "train")
        self.log("epoch_loss", total, on_epoch=True)                 # the official scheduler monitors epoch_loss_epoch
        for name, value in parts.items():
            self.log(f"epoch_loss_{name}", value, on_epoch=True)
        if batch_idx == 0:
            w = self.network.gate(torch.tensor([0, 7, 14, 28], device=self.device))
            self.log("gate_w_7", w[1], on_epoch=True, on_step=False)
            self.log("gate_w_28", w[3], on_epoch=True, on_step=False)
        return total

    def validation_step(self, batch, batch_idx):
        _, val_like, parts, _ = self._step(batch, "val")
        self.log("epoch_val_loss", val_like, on_epoch=True)         # ModelCheckpoint monitors this key, as in the official code
        for name, value in parts.items():
            self.log(f"epoch_val_loss_{name}", value, on_epoch=True)
        return val_like

    def test_step(self, batch, batch_idx):
        _, _, _, values = self._step(batch, "test")
        return [values["accuracy_test_c_overall_post"].item(), values["accuracy_test_c_post"].item(),
                values["accuracy_test_y_xy"].item(), values["accuracy_test_y_post"].item()]

    def test_epoch_end(self, outputs):
        mean = torch.tensor(outputs).mean(0)
        for name, value in zip(("accuracy_test_c_overall_post", "accuracy_test_c_post", "accuracy_test_y_xy",
                                "accuracy_test_y_post"), mean):
            logging.info("%s: %s", name, float(value))
