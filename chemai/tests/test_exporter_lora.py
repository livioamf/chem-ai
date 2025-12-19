# ruff: noqa: PLR6301, PLW1514
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from chemai.callbacks import BestModelExporter
from chemai.tests.mocks.dummy_model import DummyModel
from chemai.tests.mocks.mock_peft import MockPeftModel


class DummyBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('cfg', (), {'hidden_size': 8})

    def forward(self, input_ids, attention_mask):
        _ = attention_mask
        batch = input_ids.shape[0]
        return type('obj', (), {'last_hidden_state': torch.randn(batch, 5, 8)})


def test_exporter_with_lora(tmp_path):
    # --- prepara modelo base e PEFT
    base = DummyBaseModel()
    lora_model = MockPeftModel(base)
    model = DummyModel(lora_model)
    # cria checkpoint falso
    ckpt_path = tmp_path / 'best.ckpt'
    torch.save(model.state_dict(), ckpt_path)
    ckpt_callback = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    ckpt_callback.best_model_path = str(ckpt_path)
    # fake trainer
    trainer = Trainer(callbacks=[ckpt_callback], logger=False)
    exporter = BestModelExporter(export_dir=str(tmp_path / 'exported'))
    exporter.on_train_end(trainer, model)
    exported = Path(tmp_path / 'exported')
    # verifica merge_and_unload()
    assert (exported / 'merged.bin').exists()
    # verifica mlp
    assert (exported / 'mlp.pt').exists()
