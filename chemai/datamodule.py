import pytorch_lightning as pl
from torch.utils.data import DataLoader

from chemai.dataset import BaseSMILESDataset


class ChemBERTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        train_data,
        dev_data=None,
        test_data=None,
        batch_size=64,
        max_length=128,
    ):
        super().__init__()

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def _build_dataset(self, data):
        is_pure = 'smiles_2' not in data or data['smiles_2'] is None
        if is_pure:
            return BaseSMILESDataset(
                tokenizer=self.tokenizer,
                smiles_1=data['smiles'],
                temperatures=data['temperatures'],
                y=data['y'],
                smiles_2=None,
                frac=None,
                max_length=self.max_length,
            )
        return BaseSMILESDataset(
            tokenizer=self.tokenizer,
            smiles_1=data['smiles_1'],
            smiles_2=data['smiles_2'],
            temperatures=data['temperatures'],
            frac=data['frac'],
            y=data['y'],
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        if stage in {'fit', None}:
            self.train_ds = self._build_dataset(self.train_data)
            if self.dev_data is not None:
                self.dev_ds = self._build_dataset(self.dev_data)
        if stage == 'test':
            self.test_ds = self._build_dataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )
