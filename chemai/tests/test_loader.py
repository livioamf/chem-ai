import pandas as pd
import pytest

from chemai.loader import (
    COL_FRAC,
    COL_LOGV,
    COL_MOL,
    COL_MOL1,
    COL_MOL2,
    COL_T,
    DipprDatasetLoader,
)


# ---------------------------------------------------------
# Fixtures utilitárias
# ---------------------------------------------------------
@pytest.fixture
def sample_data_df():
    # data.csv → colunas moleculares + logV
    return pd.DataFrame({
        COL_MOL1: ['A', 'B', 'C'],
        COL_MOL2: ['D', pd.NA, 'A'],
        COL_LOGV: [1.2, 1.5, 1.0],
    })


@pytest.fixture
def sample_features_df():
    # data_features.csv → propriedades experimentais / condições
    return pd.DataFrame({
        COL_FRAC: [0.3, 1.0, 0.0],  # linha 1 → puro ; linha 2 → puro via swap
        COL_T: [300, 350, 400],
    })


@pytest.fixture
def sample_raw_df(sample_data_df, sample_features_df):
    """
    DataFrame usado diretamente em test_split_pure_mix.
    Reproduz exatamente o formato do loader.load_raw().
    """
    return pd.concat([sample_data_df, sample_features_df], axis=1).reset_index(
        drop=True
    )


@pytest.fixture
def loader(tmp_path, sample_data_df, sample_features_df):
    """Cria arquivos DIPPR simulados em disco."""
    # treinos
    sample_data_df.to_csv(tmp_path / 'data.csv', index=False)
    sample_features_df.to_csv(tmp_path / 'data_features.csv', index=False)
    # testes
    sample_data_df.to_csv(tmp_path / 'test.csv', index=False)
    sample_features_df.to_csv(tmp_path / 'test_features.csv', index=False)
    return DipprDatasetLoader(str(tmp_path))


# ---------------------------------------------------------
# Testes
# ---------------------------------------------------------
def test_load_raw(loader):
    loader.load_raw()
    assert loader.train_raw is not None
    assert loader.test_raw is not None
    expected_cols = [
        COL_MOL1,
        COL_MOL2,
        COL_FRAC,
        COL_T,
        COL_LOGV,
    ]
    assert list(loader.train_raw.columns) == expected_cols
    assert list(loader.test_raw.columns) == expected_cols


def test_normalization(sample_raw_df):
    df = DipprDatasetLoader._normalize(sample_raw_df)
    # Caso x=1 → MOL_2 deve virar NA
    assert pd.isna(df.loc[1, COL_MOL2])
    # Caso x=0 → MOL_1 deve assumir o conteúdo de MOL_2
    assert df.loc[2, COL_MOL1] == 'A'
    assert pd.isna(df.loc[2, COL_MOL2])
    assert df.loc[2, COL_FRAC] == 1.0


def test_split_pure_mix(sample_raw_df):
    df = DipprDatasetLoader._normalize(sample_raw_df)
    pure, mix = DipprDatasetLoader._split_pure_mix(df)
    # Verifica puro (x=1 e x=0 viram puros)
    expected_pure = 2
    expected_mix = 1
    assert COL_MOL in pure.columns
    assert len(pure) == expected_pure  # linhas 1 e 2 viram puros
    # Verifica misturas
    assert len(mix) == expected_mix  # linha 0 é mistura


def test_full_pipeline(loader):
    loader.prepare()
    pure = loader.get_pure()
    mix = loader.get_mix()
    assert 'train' in pure
    assert 'test' in pure
    assert 'train' in mix
    assert 'test' in mix
    assert isinstance(pure['train'], pd.DataFrame)
    assert isinstance(mix['test'], pd.DataFrame)
