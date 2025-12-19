import torch

from chemai.model import ChemBERTModel


class DummyBaseModel(torch.nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.config = type('cfg', (), {'hidden_size': hidden})

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        return type(
            'obj',
            (),
            {'last_hidden_state': torch.randn(batch_size, 8, self.config.hidden_size)},
        )


def test_model_mix_forward_symmetry():
    base = DummyBaseModel(hidden=8)
    model = ChemBERTModel(base_model=base, mode='mix')
    batch = {
        'input_ids_1': torch.ones(3, 12, dtype=torch.long),
        'attention_mask_1': torch.ones(3, 12, dtype=torch.long),
        'input_ids_2': torch.ones(3, 12, dtype=torch.long),
        'attention_mask_2': torch.ones(3, 12, dtype=torch.long),
        'temperatures': torch.tensor([300, 310, 320], dtype=torch.float),
        'frac': torch.tensor([0.2, 0.4, 0.8], dtype=torch.float),
    }
    out = model(batch)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_model_mix_backward():
    base = DummyBaseModel(hidden=8)
    model = ChemBERTModel(base_model=base, mode='mix')
    batch = {
        'input_ids_1': torch.ones(2, 12, dtype=torch.long),
        'attention_mask_1': torch.ones(2, 12, dtype=torch.long),
        'input_ids_2': torch.ones(2, 12, dtype=torch.long),
        'attention_mask_2': torch.ones(2, 12, dtype=torch.long),
        'temperatures': torch.tensor([300, 310], dtype=torch.float),
        'frac': torch.tensor([0.5, 0.7], dtype=torch.float),
        'y': torch.tensor([1.0, 2.0], dtype=torch.float),
    }
    loss = model.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
