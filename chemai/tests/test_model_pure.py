import torch

from chemai.model import ChemBERTModel


class DummyBaseModel(torch.nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.config = type('cfg', (), {'hidden_size': hidden})

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        # Simula embedding CLS no token 0
        return type(
            'obj',
            (),
            {'last_hidden_state': torch.randn(batch_size, 10, self.config.hidden_size)},
        )


def test_model_pure_forward():
    base = DummyBaseModel(hidden=16)
    model = ChemBERTModel(base_model=base, mode='pure')
    batch = {
        'input_ids_1': torch.ones(4, 10, dtype=torch.long),
        'attention_mask_1': torch.ones(4, 10, dtype=torch.long),
        'temperatures': torch.tensor([300, 310, 320, 330], dtype=torch.float),
    }
    out = model(batch)
    assert out.shape == (4,)  # saída 1D


def test_model_pure_backward():
    base = DummyBaseModel(hidden=16)
    model = ChemBERTModel(base_model=base, mode='pure')
    batch = {
        'input_ids_1': torch.ones(2, 10, dtype=torch.long),
        'attention_mask_1': torch.ones(2, 10, dtype=torch.long),
        'temperatures': torch.tensor([300, 310], dtype=torch.float),
        'y': torch.tensor([1.0, 2.0], dtype=torch.float),
    }
    loss = model.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
