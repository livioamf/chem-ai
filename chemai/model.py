import pytorch_lightning as pl
import torch
import torchmetrics
from peft import PeftModel
from torch import nn


class ChemBERTModel(pl.LightningModule):
    def __init__(
        self,
        base_model,
        mode='pure',
        hidden_dim=128,
        dropout=0.2,
        lr_head=1e-3,
        lr_lora=2e-4,
        weight_decay=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])
        self.base_model = base_model
        self.mode = mode
        self.embedding_dim = base_model.config.hidden_size

        self.is_lora = isinstance(self.base_model, PeftModel)
        if not self.is_lora:
            print('Base model congelado (train apenas MLP).')
            for p in self.base_model.parameters():
                p.requires_grad = False
        else:
            print('LoRA detectado — apenas módulos LoRA terão gradientes.')
            for name, p in self.base_model.named_parameters():
                if 'lora' in name.lower():
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        if mode == 'pure':
            mlp_in_dim = self.embedding_dim + 1  # CLS + T
        else:
            mlp_in_dim = self.embedding_dim * 2 + 2  # CLS1 + CLS2 + T + frac
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.lr_head = lr_head
        self.lr_lora = lr_lora
        self.weight_decay = weight_decay

        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()

    def forward(self, batch):
        out1 = self.base_model(
            input_ids=batch['input_ids_1'],
            attention_mask=batch['attention_mask_1'],
        )
        cls1 = out1.last_hidden_state[:, 0, :]

        if self.mode == 'pure':
            t = batch['temperatures'].unsqueeze(1).float()
            x = torch.cat([cls1, t], dim=1)
            return self.mlp(x).squeeze(1)

        out2 = self.base_model(
            input_ids=batch['input_ids_2'],
            attention_mask=batch['attention_mask_2'],
        )
        cls2 = out2.last_hidden_state[:, 0, :]
        t = batch['temperatures'].unsqueeze(1).float()
        f = batch['frac'].unsqueeze(1).float()

        x1 = torch.cat([cls1, cls2, t, f], dim=1)
        x2 = torch.cat([cls2, cls1, t, 1 - f], dim=1)
        y = 0.5 * (self.mlp(x1) + self.mlp(x2))
        return y.squeeze(1)

    def training_step(self, batch, _):
        y_hat = self(batch)
        y = batch['y']
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.train_r2.update(y_hat.detach().cpu(), y.detach().cpu())
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        r2 = self.train_r2.compute()
        self.log('train_r2', r2, prog_bar=True)
        self.train_r2.reset()

    def validation_step(self, batch, _):
        y_hat = self(batch)
        y = batch['y']
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.val_r2.update(y_hat.detach().cpu(), y.detach().cpu())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        r2 = self.val_r2.compute()
        self.log('val_r2', r2, prog_bar=True)
        self.val_r2.reset()

    def configure_optimizers(self):
        params_head = {'params': self.mlp.parameters(), 'lr': self.lr_head}
        if self.is_lora:
            lora_params = {
                'params': [p for p in self.base_model.parameters() if p.requires_grad],
                'lr': self.lr_lora,
            }
            opt = torch.optim.AdamW(
                [params_head, lora_params],
                weight_decay=self.weight_decay,
            )
        else:
            opt = torch.optim.AdamW(
                [params_head],
                weight_decay=self.weight_decay,
            )
        return opt
