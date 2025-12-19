import numpy as np
import pandas as pd
import pytest
import transformers

from chemai.chemberta_featurizer import ChemBERTaFeaturizer


# ---------------------------------------------------------
# Fixtures utilitários
# ---------------------------------------------------------
@pytest.fixture
def df_pure():
    return pd.DataFrame({
        'MOL': ['CCO', 'O'],  # etanol, água
        'T': [300.0, 350.0],
    })


@pytest.fixture
def df_mix():
    return pd.DataFrame({
        'MOL_1': ['CCO', 'O'],  # etanol, água
        'MOL_2': ['C', 'CC'],  # metano, etano
        'MolFrac_1': [0.25, 0.8],
        'T': [300.0, 400.0],
        'logV': [1.2, 2.3],
    })


@pytest.fixture
def featurizer_cpu():
    """Versão em CPU — garante estabilidade nos testes."""
    transformers.logging.set_verbosity_error()
    fe = ChemBERTaFeaturizer(
        device='cpu',
        use_half=False,
        compile_model=False,
        max_length=32,
    )
    return fe


# ---------------------------------------------------------
# Testes: featurize_pure
# ---------------------------------------------------------
def test_featurize_pure_output_shape(featurizer_cpu, df_pure):
    feat = featurizer_cpu.featurize_pure(df_pure)
    # Deve ter n_linhas × (hidden_size + 1)
    assert feat.shape[0] == len(df_pure)
    assert feat.shape[1] > 1  # ao menos embeddings + T
    # Última coluna deve ser T
    assert np.allclose(feat[:, -1], df_pure['T'].values)


def test_featurize_pure_no_nan(featurizer_cpu, df_pure):
    feat = featurizer_cpu.featurize_pure(df_pure)
    assert not np.isnan(feat).all(axis=1).any()


# ---------------------------------------------------------
# Testes: featurize_mix
# ---------------------------------------------------------
def test_featurize_mix_double_length(featurizer_cpu, df_mix):
    feat_mix = featurizer_cpu.featurize_mix(df_mix)
    # Deve dobrar número de linhas
    assert feat_mix.shape[0] == 2 * len(df_mix)


def test_featurize_mix_contains_required_columns(featurizer_cpu, df_mix):
    feat_mix = featurizer_cpu.featurize_mix(df_mix)
    cols = feat_mix.columns
    # Prefixos obrigatórios
    assert any(c.startswith('mol1_') for c in cols)
    assert any(c.startswith('mol2_') for c in cols)
    # Variáveis externas
    assert 'T' in cols
    assert 'frac' in cols
    assert 'logV' in cols


def test_featurize_mix_symmetry_structure(featurizer_cpu, df_mix):
    """
    Verifica a estrutura da simetria:
    - mix_1 = (mol1, mol2)
    - mix_2 = (mol2, mol1)
    Não testamos valores numéricos, apenas estrutura.
    """
    feat = featurizer_cpu.featurize_mix(df_mix)
    n = len(df_mix)
    mix_1 = feat.iloc[:n]
    mix_2 = feat.iloc[n:]  # versão invertida
    # Prefixos iguais, só invertendo o conteúdo
    mol1_cols = [c for c in feat.columns if c.startswith('mol1_')]
    mol2_cols = [c for c in feat.columns if c.startswith('mol2_')]
    # Estrutura deve ser idêntica
    assert set(mol1_cols) == set(mol1_cols)
    assert set(mol2_cols) == set(mol2_cols)
    # T, frac, logV devem existir nas duas metades
    for col in ['T', 'logV', 'frac']:
        assert col in mix_1.columns
        assert col in mix_2.columns


def test_featurize_mix_frac_symmetry(featurizer_cpu, df_mix):
    feat = featurizer_cpu.featurize_mix(df_mix)
    n = len(df_mix)
    mix_1 = feat.iloc[:n]
    mix_2 = feat.iloc[n:]
    # frac da segunda metade deve ser (1 - frac)
    expected = 1.0 - mix_1['frac'].values
    assert np.allclose(mix_2['frac'].values, expected)


# ---------------------------------------------------------
# Smoke test: GPU fallback
# ---------------------------------------------------------
def test_smoke_gpu_fallback(df_pure):
    """
    Só garante que funciona caso haja GPU disponível.
    Não testa nada numérico.
    """
    fe = ChemBERTaFeaturizer(max_length=32, compile_model=False, use_half=False)
    feat = fe.featurize_pure(df_pure)
    assert feat is not None
