import pytorch_lightning as pl
import torch
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader

from chemai.datamodule import ChemBERTDataModule
from chemai.dataset import BaseSMILESDataset
from chemai.model import ChemBERTModel


# ======================================================================
# 1. TOKENIZER FAKE (sem HuggingFace)
# ======================================================================
class DummyTokenizer:
    def __call__(self, smiles, padding, truncation, max_length):
        _ = padding
        _ = truncation
        return {
            'input_ids': [[1] * max_length for _ in smiles],
            'attention_mask': [[1] * max_length for _ in smiles],
        }


# ======================================================================
# 2. MODELO FAKE HUGGINGFACE
# ======================================================================
class DummyBaseHF(torch.nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = type('cfg', (), {'hidden_size': hidden_size})

    def forward(self, input_ids, attention_mask):
        batch = input_ids.size(0)
        L = input_ids.size(1)
        hidden = torch.randn(batch, L, self.config.hidden_size)
        return type('obj', (), {'last_hidden_state': hidden})


# ======================================================================
# 3. PEFT MODEL FAKE (LoRA)
# ======================================================================
class DummyPeftModel(PeftModel):
    """
    Dummy PEFT model:
    - permite isinstance(..., PeftModel)
    - mantém base_model como submódulo PyTorch válido
    - cria parâmetros LoRA fictícios
    - NÃO usa lógica real de PEFT
    """

    def __init__(self, base_model):
        # Inicializa corretamente como nn.Module
        nn.Module.__init__(self)
        # Armazena base_model como submódulo
        self.base_model = base_model
        self.config = base_model.config
        # Cria parâmetros LoRA FAKE treináveis
        self.lora_layer = nn.Linear(base_model.config.hidden_size, 4)
        self.lora_layer.weight.requires_grad = True
        self.lora_layer.bias.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # simplesmente passa para o modelo base
        return self.base_model(input_ids, attention_mask)

    def parameters(self, recurse=True):
        _ = recurse
        # Somente retorna os parâmetros LoRA
        return self.lora_layer.parameters()


# ======================================================================
# FIXTURES
# ======================================================================
def make_pure_data():
    return {
        'smiles_1': ['CCO', 'O', 'CCC'],
        'temperatures': [300, 320, 340],
        'y': [1.0, 1.5, 2.0],
    }


def make_mix_data():
    return {
        'smiles_1': ['CCO', 'O'],
        'smiles_2': ['C', 'CC'],
        'temperatures': [300, 330],
        'frac': [0.2, 0.8],
        'y': [1.0, 2.0],
    }


# ======================================================================
# 4. INTEGRAÇÃO: PURE
# ======================================================================
def test_integration_pure_forward_backward():
    base = DummyBaseHF(hidden_size=12)
    model = ChemBERTModel(base, mode='pure')
    data = make_pure_data()
    ds = BaseSMILESDataset(
        tokenizer=DummyTokenizer(),
        smiles_1=data['smiles_1'],
        temperatures=data['temperatures'],
        y=data['y'],
        max_length=16,
    )
    batch = next(iter(DataLoader(ds, batch_size=2)))
    # forward
    out = model(batch)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # backward
    loss = torch.nn.functional.mse_loss(out, batch['y'])
    loss.backward()
    assert torch.isfinite(loss)


# ======================================================================
# 5. INTEGRAÇÃO: MIX (simetria)
# ======================================================================
def test_integration_mix_symmetry():
    base = DummyBaseHF(hidden_size=10)
    model = ChemBERTModel(base, mode='mix')
    data = make_mix_data()
    ds = BaseSMILESDataset(
        tokenizer=DummyTokenizer(),
        smiles_1=data['smiles_1'],
        smiles_2=data['smiles_2'],
        temperatures=data['temperatures'],
        frac=data['frac'],
        y=data['y'],
        max_length=12,
    )
    batch = next(iter(DataLoader(ds, batch_size=2)))
    out = model(batch)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


# ======================================================================
# 6. TRAINER (1 passo) — PURE
# ======================================================================
def test_trainer_step_pure():
    base = DummyBaseHF(hidden_size=8)
    model = ChemBERTModel(base, mode='pure')
    train = make_pure_data()
    dev = make_pure_data()
    dm = ChemBERTDataModule(
        DummyTokenizer(), train_data=train, dev_data=dev, batch_size=2
    )
    dm.setup()
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, dm)


# ======================================================================
# 7. TRAINER (1 passo) — MIX
# ======================================================================
def test_trainer_step_mix():
    base = DummyBaseHF(hidden_size=8)
    model = ChemBERTModel(base, mode='mix')
    train = make_mix_data()
    dev = make_mix_data()
    dm = ChemBERTDataModule(
        DummyTokenizer(), train_data=train, dev_data=dev, batch_size=2
    )
    dm.setup()
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, dm)


# ======================================================================
# 8. LoRA / PEFT — integração
# ======================================================================
def test_integration_lora_forward_backward():
    base = DummyBaseHF(hidden_size=12)
    lora = DummyPeftModel(base)
    model = ChemBERTModel(lora, mode='pure', lr_lora=5e-4)
    data = make_pure_data()
    ds = BaseSMILESDataset(
        tokenizer=DummyTokenizer(),
        smiles_1=data['smiles_1'],
        temperatures=data['temperatures'],
        y=data['y'],
        max_length=16,
    )
    batch = next(iter(DataLoader(ds, batch_size=2)))
    out = model(batch)
    loss = torch.nn.functional.mse_loss(out, batch['y'])
    loss.backward()
    # Apenas parâmetros LoRA devem ter gradientes
    for name, param in model.base_model.__dict__.items():
        if isinstance(param, torch.nn.Parameter):
            if 'lora' in name.lower():
                assert param.grad is not None
            else:
                assert param.grad is None
