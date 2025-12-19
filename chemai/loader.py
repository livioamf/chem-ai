import numpy as np
import pandas as pd
from pandas import DataFrame

COL_MOL1 = 'MOL_1'
COL_MOL2 = 'MOL_2'
COL_FRAC = 'MolFrac_1'
COL_T = 'T'
COL_LOGV = 'logV'
COL_MOL = 'MOL'


class DipprDatasetLoader:
    def __init__(self, data_dir: str = '../data/nist_dippr_data'):
        self.data_dir = data_dir

        self.train_raw: DataFrame | None = None
        self.test_raw: DataFrame | None = None
        self.train_norm: DataFrame | None = None
        self.test_norm: DataFrame | None = None
        self.pure: dict[str, DataFrame] = {}
        self.mix: dict[str, DataFrame] = {}

    @staticmethod
    def _normalize(df: DataFrame) -> DataFrame:
        d = df.copy()
        d[COL_MOL1] = d[COL_MOL1].astype('string')
        d[COL_MOL2] = d[COL_MOL2].astype('string')
        x = d[COL_FRAC].astype(float)
        tol = 1e-8
        mask_1 = np.isclose(x, 1.0, atol=tol, rtol=0.0)
        mask_0 = np.isclose(x, 0.0, atol=tol, rtol=0.0)
        d.loc[mask_1, COL_MOL2] = pd.NA
        d.loc[mask_0, COL_MOL1] = d.loc[mask_0, COL_MOL2].values
        d.loc[mask_0, COL_MOL2] = pd.NA
        d.loc[mask_0, COL_FRAC] = 1.0
        mask_sort = d[COL_MOL2].notnull() & (d[COL_MOL1] > d[COL_MOL2])
        tmp = d.loc[mask_sort, COL_MOL1].copy().values
        d.loc[mask_sort, COL_MOL1] = d.loc[mask_sort, COL_MOL2]
        d.loc[mask_sort, COL_MOL2] = tmp
        d.loc[mask_sort, COL_FRAC] = 1.0 - d.loc[mask_sort, COL_FRAC].values
        return d

    @staticmethod
    def _split_pure_mix(df: DataFrame) -> tuple[DataFrame, DataFrame]:
        pure = (
            df[df[COL_MOL2].isna()]
            .reset_index(drop=True)[[COL_MOL1, COL_T, COL_LOGV]]
            .rename(columns={COL_MOL1: COL_MOL})
        )
        mix = df[df[COL_MOL2].notna()].reset_index(drop=True)
        return pure, mix

    def load_raw(self) -> None:
        train_data = pd.read_csv(f'{self.data_dir}/data.csv')
        train_feat = pd.read_csv(f'{self.data_dir}/data_features.csv')
        self.train_raw = (
            pd.concat([train_feat, train_data], axis=1)
            .dropna()
            .reset_index(drop=True)[[COL_MOL1, COL_MOL2, COL_FRAC, COL_T, COL_LOGV]]
        )
        test_data = pd.read_csv(f'{self.data_dir}/test.csv')
        test_feat = pd.read_csv(f'{self.data_dir}/test_features.csv')
        self.test_raw = (
            pd.concat([test_feat, test_data], axis=1)
            .dropna()
            .reset_index(drop=True)[[COL_MOL1, COL_MOL2, COL_FRAC, COL_T, COL_LOGV]]
        )

    def prepare(self) -> None:
        if self.train_raw is None or self.test_raw is None:
            self.load_raw()
        self.train_norm = self._normalize(self.train_raw)
        self.test_norm = self._normalize(self.test_raw)
        train_pure, train_mix = self._split_pure_mix(self.train_norm)
        test_pure, test_mix = self._split_pure_mix(self.test_norm)
        self.pure = {'train': train_pure, 'test': test_pure}
        self.mix = {'train': train_mix, 'test': test_mix}

    def get_pure(self) -> dict[str, DataFrame]:
        if not self.pure:
            self.prepare()
        return self.pure

    def get_mix(self) -> dict[str, DataFrame]:
        if not self.mix:
            self.prepare()
        return self.mix
