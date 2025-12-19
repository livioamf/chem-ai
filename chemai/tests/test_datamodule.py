from chemai.datamodule import ChemBERTDataModule
from chemai.dataset import BaseSMILESDataset


class DummyTokenizer:
    def __call__(self, smiles, padding, truncation, max_length):
        return {
            'input_ids': [[1] * max_length for _ in smiles],
            'attention_mask': [[1] * max_length for _ in smiles],
        }


def test_datamodule_pure():
    train = {
        'smiles_1': ['CCO', 'O'],
        'temperatures': [300, 350],
        'y': [1.0, 1.5],
    }
    dev = {
        'smiles_1': ['CCC'],
        'temperatures': [320],
        'y': [1.2],
    }
    dm = ChemBERTDataModule(
        DummyTokenizer(), train_data=train, dev_data=dev, batch_size=2
    )
    dm.setup()
    assert isinstance(dm.train_ds, BaseSMILESDataset)
    assert not dm.train_ds.has_smiles2
    assert dm.train_ds.y is not None
    train_batch = next(iter(dm.train_dataloader()))
    assert 'input_ids_1' in train_batch
    assert 'temperatures' in train_batch


def test_datamodule_mix():
    train = {
        'smiles_1': ['CCO', 'O'],
        'smiles_2': ['C', 'CC'],
        'temperatures': [300, 350],
        'frac': [0.1, 0.9],
        'y': [1.0, 2.0],
    }
    dev = train
    dm = ChemBERTDataModule(
        DummyTokenizer(), train_data=train, dev_data=dev, batch_size=2
    )
    dm.setup()
    assert dm.train_ds.has_smiles2
    assert 'frac' in dm.train_ds[0]
