# ruff: noqa: PLR6301, PLW1514
from peft import PeftModel
from torch import nn


class MockMergedBase(nn.Module):
    """Modelo resultante de merge_and_unload (simples HF format)."""

    def save_pretrained(self, export_dir):
        open(f'{export_dir}/merged.bin', 'w', encoding='utf-8').write('OK')


class MockPeftModel(PeftModel):
    """
    Mock simples que imita comportamento do PEFT para testes:
    - possui merge_and_unload()
    - forward delega ao modelo base
    """

    def __init__(self, base_model):
        nn.Module.__init__(self)  # IMPORTANTE
        self.base_model = base_model
        self.config = base_model.config
        # parâmetros LoRA fictícios
        self.lora = nn.Linear(base_model.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        return self.base_model(input_ids, attention_mask)

    # permite que o model exporter detecte e mescle
    def merge_and_unload(self):
        return MockMergedBase()
