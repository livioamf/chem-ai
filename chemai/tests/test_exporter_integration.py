# ruff: noqa: PLR6301, PLW1514

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from chemai.callbacks import BestModelExporter
from chemai.tests.mocks.dummy_model import DummyModel


def test_exporter_full_integration(tmp_path):
    model = DummyModel()
    ckpt_callback = ModelCheckpoint(
        dirpath=tmp_path,
        save_last=True,
        save_top_k=0,  # <── desnecessário monitor
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=True,
        callbacks=[ckpt_callback],
    )
    trainer.fit(model)
    exporter = BestModelExporter(export_dir=str(tmp_path / 'final_export'))
    exporter.on_train_end(trainer, model)
    export_dir = tmp_path / 'final_export'
    # MLP deve ter sido salva
    assert (export_dir / 'mlp.pt').exists()
