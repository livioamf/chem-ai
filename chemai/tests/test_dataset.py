from chemai.dataset import BaseSMILESDataset


class DummyTokenizer:
    """
    Tokenizer fake, rápido, sem dependências externas.
    """

    def __call__(self, smiles, padding, truncation, max_length):
        _ = padding
        _ = truncation
        batch_size = len(smiles)
        return {
            'input_ids': [[1] * max_length for _ in range(batch_size)],
            'attention_mask': [[1] * max_length for _ in range(batch_size)],
        }


def test_dataset_pure():
    tok = DummyTokenizer()
    smiles = ['CCO', 'O']
    temps = [300, 350]
    y = [1.0, 2.0]
    ds = BaseSMILESDataset(tok, smiles_1=smiles, temperatures=temps, y=y)
    assert len(ds) == len(smiles)
    sample = ds[0]
    assert 'input_ids_1' in sample
    assert 'temperatures' in sample
    assert 'y' in sample
    assert 'input_ids_2' not in sample  # PURE


def test_dataset_mix():
    tok = DummyTokenizer()
    test_t = 300
    ds = BaseSMILESDataset(
        tok,
        smiles_1=['CCO', 'O'],
        smiles_2=['C', 'CC'],
        temperatures=[test_t, 400],
        frac=[0.2, 0.8],
        y=[1.1, 2.2],
    )
    sample = ds[0]
    assert 'input_ids_2' in sample  # MIX
    assert 'frac' in sample
    assert sample['temperatures'] == test_t
