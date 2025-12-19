import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from chemai.chem_featurizer import ChemFeaturizer

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------


@pytest.fixture
def featurizer():
    return ChemFeaturizer()


@pytest.fixture
def df_pure():
    # MOL, T, logV
    return pd.DataFrame({
        'MOL': ['CCO', 'O'],  # etanol, água
        'T': [300.0, 350.0],
        'logV': [1.23, 0.98],
    })


@pytest.fixture
def df_mix():
    # MOL_1, MOL_2, MolFrac_1, T, logV
    return pd.DataFrame({
        'MOL_1': ['CCO', 'O'],  # etanol / água
        'MOL_2': ['C', 'CC'],  # metano / etano
        'MolFrac_1': [0.25, 0.8],
        'T': [300.0, 400.0],
        'logV': [1.2, 2.3],
    })


# ---------------------------------------------------------
# Testa smiles_to_mol
# ---------------------------------------------------------
def test_smiles_to_mol(featurizer):
    smiles = ['CCO', 'O']
    mols = featurizer.smiles_to_mol(smiles)
    assert len(mols) == len(smiles)
    assert all(isinstance(m, Chem.Mol) for m in mols)
    assert mols[0].GetNumAtoms() > 0


# ---------------------------------------------------------
# Testa get_features
# ---------------------------------------------------------
def test_get_features(featurizer):
    mol = Chem.MolFromSmiles('CCO')
    feats = featurizer.get_features(mol)
    # Checa se retornou um dict com floats
    assert isinstance(feats, dict)
    assert 'mw' in feats
    assert isinstance(feats['mw'], float)


# ---------------------------------------------------------
# Testa featurize_pure (com paralelização)
# ---------------------------------------------------------
def test_featurize_pure(featurizer, df_pure):
    feat_df = featurizer.featurize_pure(df_pure, n_jobs=2)
    # Mesmo número de linhas
    assert len(feat_df) == len(df_pure)
    # Verificar colunas essenciais
    assert 'mw' in feat_df.columns
    assert 'T' in feat_df.columns
    assert 'logV' in feat_df.columns
    # Checar se valores de T foram copiados
    assert np.allclose(feat_df['T'].values, df_pure['T'].values)


# ---------------------------------------------------------
# Testa featurize_mix_parallel
# ---------------------------------------------------------
def test_featurize_mix_parallel(featurizer, df_mix):
    feat_mix = featurizer.featurize_mix_parallel(df_mix, n_jobs=2)
    # O número de linhas deve dobrar
    assert len(feat_mix) == 2 * len(df_mix)
    # Checar colunas de mol1 e mol2
    assert any(col.startswith('mol1_') for col in feat_mix.columns)
    assert any(col.startswith('mol2_') for col in feat_mix.columns)
    # Verificar colunas externas
    assert 'T' in feat_mix.columns
    assert 'frac' in feat_mix.columns
    assert 'logV' in feat_mix.columns
    # Verificar consistência dos valores de T
    assert set(feat_mix['T'].values) == set(df_mix['T'].values)
    # Verificar que frac contém MolFrac_1 e 1 - MolFrac_1
    orig = df_mix['MolFrac_1'].values
    expected_fracs = set(orig) | set(1 - orig)
    assert set(feat_mix['frac'].values) == expected_fracs
