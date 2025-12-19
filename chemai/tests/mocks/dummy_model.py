import pytorch_lightning as pl
import torch


class DummyModel(pl.LightningModule):
    def __init__(self, base_model=None):
        super().__init__()
        # Permite uso tanto no trainer.fit quanto nos mocks
        self.base = base_model or torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Linear(4, 1)
        # batch fake
        self._batch = {'x': torch.randn(1, 4), 'y': torch.randn(1)}

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, base_model):
        # Recria o modelo e injeta o backbone
        return cls(base_model=base_model)

    def training_step(self, batch, batch_idx):
        # Loss precisa ter grad_fn
        fake_input = torch.randn(1, 4)
        return self.mlp(fake_input).sum()

    def train_dataloader(self):
        return torch.utils.data.DataLoader([self._batch], batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.mlp.parameters(), lr=1e-3)
