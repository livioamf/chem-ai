import torch
from torch.utils.data import Dataset


class BaseSMILESDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        smiles_1,
        temperatures,
        smiles_2=None,
        frac=None,
        y=None,
        max_length=128,
    ):
        enc1 = tokenizer(
            smiles_1,
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )
        self.input_ids_1 = torch.tensor(enc1['input_ids'], dtype=torch.long)
        self.att_mask_1 = torch.tensor(enc1['attention_mask'], dtype=torch.long)

        self.has_smiles2 = smiles_2 is not None
        if self.has_smiles2:
            enc2 = tokenizer(
                smiles_2,
                padding='max_length',
                truncation=True,
                max_length=max_length,
            )
            self.input_ids_2 = torch.tensor(enc2['input_ids'], dtype=torch.long)
            self.att_mask_2 = torch.tensor(enc2['attention_mask'], dtype=torch.long)
        self.temperatures = torch.tensor(temperatures, dtype=torch.float)
        self.frac = None if frac is None else torch.tensor(frac, dtype=torch.float)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.input_ids_1)

    def __getitem__(self, idx):
        item = {
            'input_ids_1': self.input_ids_1[idx],
            'attention_mask_1': self.att_mask_1[idx],
            'temperatures': self.temperatures[idx],
        }
        if self.has_smiles2:
            item['input_ids_2'] = self.input_ids_2[idx]
            item['attention_mask_2'] = self.att_mask_2[idx]
        if self.frac is not None:
            item['frac'] = self.frac[idx]
        if self.y is not None:
            item['y'] = self.y[idx]
        return item
