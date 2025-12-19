# ruff: noqa: PLR6301, PLW1514
import os
from unittest.mock import MagicMock

from pytorch_lightning.callbacks import ModelCheckpoint

from chemai.callbacks import BestModelExporter
from chemai.tests.mocks.dummy_model import DummyModel


class DummyBaseModel:
    def save_pretrained(self, path):
        with open(os.path.join(path, 'dummy.bin'), 'w', encoding='utf-8') as f:
            f.write('ok')


def test_exporter(tmp_path):
    ckpt_callback = ModelCheckpoint(dirpath=tmp_path, save_top_k=1)
    ckpt_path = tmp_path / 'best.ckpt'
    with open(ckpt_path, 'w', encoding='utf-8') as f:
        f.write('fake checkpoint')
    ckpt_callback.best_model_path = str(ckpt_path)
    trainer = MagicMock()
    trainer.callbacks = [ckpt_callback]
    pl_module = DummyModel(base_model=DummyBaseModel())

    class DummyTokenizer:
        def save_pretrained(self, path):
            with open(os.path.join(path, 'tokenizer.txt'), 'w', encoding='utf-8') as f:
                f.write('tok')

    tok = DummyTokenizer()
    exporter = BestModelExporter(export_dir=tmp_path / 'exported', tokenizer=tok)
    exporter.on_train_end(trainer, pl_module)
    export_dir = tmp_path / 'exported'
    # ✔ Modelo base salvo
    assert (export_dir / 'dummy.bin').exists()
    # ✔ MLP salva
    assert (export_dir / 'mlp.pt').exists()
    # ✔ Tokenizer salvo
    assert (export_dir / 'tokenizer.txt').exists()
