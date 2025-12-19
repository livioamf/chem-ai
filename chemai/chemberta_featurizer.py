from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles = smiles_list

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx]


def collate_fn(tokenizer, batch_smiles, max_length):
    return tokenizer(
        batch_smiles,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt',
    )


class ChemBERTaFeaturizer:
    def __init__(
        self,
        model_name='DeepChem/ChemBERTa-77M-MTR',
        device=None,
        max_length=128,
        use_half=False,
        compile_model=False,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        if use_half and torch.cuda.is_available():
            model = model.half()

        if compile_model and self.device.type == 'cuda':
            try:
                model = torch.compile(model)
            except Exception:
                print('[WARN] torch.compile desativado — Triton não encontrado.')

        model = model.to(self.device)
        model.eval()

        if compile_model and torch.cuda.is_available():
            model = torch.compile(model)
        self.model = model
        self.hidden = model.config.hidden_size

    def featurize(self, smiles_list, batch_size=512, num_workers=2):
        dataset = SmilesDataset(smiles_list)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=partial(collate_fn, self.tokenizer, max_length=self.max_length),
        )
        all_cls, all_mean = [], []
        with torch.no_grad():
            for batch in loader:
                batch_to_device = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }
                outputs = self.model(**batch_to_device)
                hidden = outputs.last_hidden_state
                cls_emb = hidden[:, 0, :]
                mean_emb = hidden.mean(dim=1)
                all_cls.append(cls_emb.cpu())
                all_mean.append(mean_emb.cpu())
                emb_cls = torch.cat(all_cls, axis=0).numpy()
                emb_mean = torch.cat(all_mean, axis=0).numpy()
        return emb_cls, emb_mean

    def featurize_pure(self, df_pure):
        smiles = df_pure['MOL'].tolist()
        emb_cls, _ = self.featurize(smiles)
        df_emb = pd.DataFrame(emb_cls).add_prefix('emb_')
        df_emb['T'] = df_pure['T'].values
        df_emb['logV'] = df_pure['logV'].values
        return df_emb

    def featurize_mix(self, df_mix):
        smiles1 = df_mix['MOL_1'].tolist()
        smiles2 = df_mix['MOL_2'].tolist()

        emb1, _ = self.featurize(smiles1)
        emb2, _ = self.featurize(smiles2)
        df_emb1 = pd.DataFrame(emb1).add_prefix('mol1_')
        df_emb2 = pd.DataFrame(emb2).add_prefix('mol2_')

        mix_1 = pd.concat([df_emb1, df_emb2], axis=1)
        mix_1['frac'] = df_mix['MolFrac_1'].values
        mix_1['T'] = df_mix['T'].values
        mix_1['logV'] = df_mix['logV'].values

        df_emb1_inv = df_emb2.copy()
        df_emb2_inv = df_emb1.copy()
        df_emb1_inv.columns = df_emb1.columns
        df_emb2_inv.columns = df_emb2.columns
        mix_2 = pd.concat([df_emb1_inv, df_emb2_inv], axis=1)
        mix_2['frac'] = 1.0 - df_mix['MolFrac_1'].values
        mix_2['T'] = df_mix['T'].values
        mix_2['logV'] = df_mix['logV'].values
        mix_features = pd.concat([mix_1, mix_2], axis=0).reset_index(drop=True)

        return mix_features
