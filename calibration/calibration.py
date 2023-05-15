import copy
import math
from typing import Dict, Optional, Tuple, Union

import fire
import numpy as np
import opacus
import torch
import transformers
from ml_swissknife import utils
from opacus.grad_sample import register_grad_sampler
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from netcal.metrics import ECE

from calibration.common import ECELoss

class TuningMethod(utils.ContainerMeta):
    dpsgd = "dpsgd"
    dpadam = "dpadam"

class TemperatureModule(nn.Module):
    def __init__(self,):
        super(TemperatureModule, self).__init__()
        self.param = nn.Parameter(torch.full(size=(), fill_value=math.log(math.e - 1), device=device), requires_grad=True)
        self.device = device

    def forward(self, x):
        return x * self.inv_temp

    @property
    def inv_temp(self):
        return F.softplus(self.param)

    @property
    def temp(self):
        return self.inv_temp ** -1.

@register_grad_sampler(TemperatureModule)
def compute_temperature_scaling_grad_sample(
    layer: TemperatureModule, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    return {layer.param: torch.sigmoid(layer.param) * (activations * backprops).sum(dim=1)}

class Trainer(object):
    def __init__(self, data, lr, momentum, epochs) -> None:
        logits, labels = data['train']
        test_logits, test_labels = data['test']
        self.train_dataset = TensorDataset(logits, labels)
        self.test_dataset = TensorDataset(test_logits, test_labels)
        self.epochs = epochs

        self.temperature_module = TemperatureModule().to(device)
        self.optimizer = optim.SGD(self.temperature_module.parameters(), lr=lr, momentum=momentum)
        train_size, test_size = len(self.train_dataset), len(self.test_dataset)
        self.train_size = train_size
        self.batch_size = train_size
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=test_size, shuffle=True)

    def train(self, target_epsilon, epochs, max_grad_norm, dp=True):
        if dp:
            print("Using DP-SGD")
            privacy_engine = opacus.PrivacyEngine()
            delta = self.train_size ** -1.1
            self.temperature_module, self.optimizer, self.train_loader = privacy_engine.make_private_with_epsilon(
                module=self.temperature_module,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                target_delta=delta,
                target_epsilon=target_epsilon,
                max_grad_norm=max_grad_norm,
                epochs=epochs,
            )
        self.lr_scheduler = transformers.get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=0, num_training_steps=self.epochs * self.train_size // self.batch_size,
        )
        for epoch in range(1, epochs + 1):
            for logits, labels in self.train_loader:
                logits, labels = logits.to(device), labels.to(device, non_blocking=True)
                self.optimizer.zero_grad()
                nll_criterion = nn.CrossEntropyLoss().to(device)
                loss = nll_criterion(self.temperature_module(logits), labels)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                train_ece = self.eval(logits, labels)

            for logits, labels in self.test_loader:
                test_ece = self.eval(logits, labels)
            print((f"Epoch {epoch}, Train NLL {loss.item():.4f}, Train ECE {train_ece.item():.4f}, Test ECE {test_ece.item():.4f}, lr {self.lr_scheduler.get_last_lr()[0]:.4f}"))
                
    def eval(self, raw_logits, labels):
        with torch.no_grad():
            raw_logits, labels = raw_logits.to(device), labels.to(device, non_blocking=True)
            logits = self.temperature_module(raw_logits)
            ece = ECELoss(n_bins=10)
            ece_loss = ece(logits, labels)
        return ece_loss

def main(
    lr=1e-1,
    lr_decay=False,
    batch_size=None, 
    max_grad_norm=1.,
    target_epsilon=8.,
    epochs=20,
    tuning_method="dpsgd",
    weight_decay=0.,
    momentum=0.0,
    dp=True,
):
    TensorOrArray = Union[np.array, torch.Tensor]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("hyeperparameters", locals())
    # NOTE save the logits and labels as npy files for the recalibration stage.
    valid_logits = np.load(VALID_LOGITS_NPYFILE)
    valid_labels = np.load(VALID_LABELS_NPYFILE)
    test_logits = np.load(TEST_LOGITS_NPYFILE)
    test_labels = np.load(TEST_LABELS_NPYFILE)

    print('data shape:')
    print(valid_logits.shape, valid_labels.shape, test_logits.shape, test_labels.shape)

    valid_logits, valid_labels, test_logits, test_labels = torch.from_numpy(valid_logits), torch.from_numpy(valid_labels), torch.from_numpy(test_logits), torch.from_numpy(test_labels)
    data = {"train": (valid_logits, valid_labels), "test": (test_logits, test_labels)}
    trainer = Trainer(data, lr, momentum, epochs)
    trainer.train(target_epsilon, epochs, max_grad_norm, dp)


if __name__ == "__main__":
    fire.Fire(main)