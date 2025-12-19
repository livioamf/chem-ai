# -*- coding: utf-8 -*-
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem import GraphDescriptors as GD


class ChemFeaturizer:
    SMARTS = {
        'has_alcohol': Chem.MolFromSmarts('[CX4;H1,H2,H3][OX2H]'),
        'has_acid': Chem.MolFromSmarts('C(=O)[OX2H]'),
        'has_amide': Chem.MolFromSmarts('C(=O)N'),
        'has_amine': Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),
        'has_urea': Chem.MolFromSmarts('NC(=O)N'),
        'has_nitrile': Chem.MolFromSmarts('C#N'),
        'has_ether': Chem.MolFromSmarts('[$([CX4]-O-[CX4])]'),
    }

    HALOGENS = {9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}

    def __init__(self):
        pass

    @staticmethod
    def smiles_to_mol(smiles_list):
        return [Chem.MolFromSmiles(s) for s in smiles_list]

    @staticmethod
    def count_halogens(mol):
        cnt = {'F': 0, 'Cl': 0, 'Br': 0, 'I': 0}
        for a in mol.GetAtoms():
            z = a.GetAtomicNum()
            if z in ChemFeaturizer.HALOGENS:
                cnt[ChemFeaturizer.HALOGENS[z]] += 1
        return cnt

    @staticmethod
    def bool_feature(mol, patt):
        return int(mol.HasSubstructMatch(patt))

    @staticmethod
    def compute_descriptor_family(mol, prefix):
        fam = {}
        for name in dir(Descriptors):
            if name.startswith(prefix):
                func = getattr(Descriptors, name, None)
                if callable(func):
                    try:
                        fam[name] = float(func(mol))
                    except Exception:
                        fam[name] = float('nan')
        return fam

    @staticmethod
    def compute_chi_descriptors(mol):
        chi = {}
        for name in ['Chi0', 'Chi1', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v']:
            f = getattr(GD, name, None)
            try:
                chi[name] = float(f(mol)) if callable(f) else float('nan')
            except Exception:
                chi[name] = float('nan')
        return chi

    @staticmethod
    def get_features(mol):  # noqa: PLR0915
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass

        feats = {}
        # Básicos/polaridade
        feats['peso_molecular'] = Descriptors.MolWt(mol)
        feats['peso_molecular_heavy'] = Descriptors.HeavyAtomMolWt(mol)
        feats['peso_molecular_exato'] = Descriptors.ExactMolWt(mol)
        feats['atomos_pesados'] = Descriptors.HeavyAtomCount(mol)
        feats['eletrons_valencia'] = Descriptors.NumValenceElectrons(mol)
        f_rad = getattr(Descriptors, 'NumRadicalElectrons', None)
        feats['eletrons_radicais'] = float(f_rad(mol))
        feats['tpsa'] = rdMolDescriptors.CalcTPSA(mol)
        feats['logp'] = Crippen.MolLogP(mol)
        feats['molar_refractivity'] = Crippen.MolMR(mol)
        feats['lig_rotacionais'] = Descriptors.NumRotatableBonds(mol)
        feats['frac_csp3'] = Descriptors.FractionCSP3(mol)

        # Forma/estrutura
        feats['hall_kier_alpha'] = Descriptors.HallKierAlpha(mol)
        feats['kappa1'] = Descriptors.Kappa1(mol)
        feats['kappa2'] = Descriptors.Kappa2(mol)
        feats['kappa3'] = Descriptors.Kappa3(mol)
        f_balaban = getattr(Descriptors, 'BalabanJ', None)
        f_bertz = getattr(Descriptors, 'BertzCT', None)
        if callable(f_balaban):
            feats['balabanJ'] = float(f_balaban(mol))
        else:
            float('nan')
        if callable(f_bertz):
            feats['bertzCT'] = float(f_bertz(mol))
        else:
            float('nan')
        feats.update(ChemFeaturizer.compute_chi_descriptors(mol))

        # Anéis
        feats['num_rings'] = rdMolDescriptors.CalcNumRings(mol)
        feats['num_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        feats['num_aliphatic_rings'] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        feats['num_saturated_rings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        feats['num_aromatic_carbocycles'] = rdMolDescriptors.CalcNumAromaticCarbocycles(
            mol
        )
        feats['num_aromatic_heterocycles'] = (
            rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        )
        feats['num_saturated_carbocycles'] = (
            rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
        )
        feats['num_saturated_heterocycles'] = (
            rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)
        )
        feats['num_aliphatic_carbocycles'] = (
            rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
        )
        feats['num_aliphatic_heterocycles'] = (
            rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
        )

        # Heteroátomos e carga formal
        feats['num_heteroatoms'] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        feats['formal_charge'] = sum(a.GetFormalCharge() for a in mol.GetAtoms())

        # Cargas parciais
        try:
            feats['max_abs_partial_charge'] = Descriptors.MaxAbsPartialCharge(mol)
            feats['min_abs_partial_charge'] = Descriptors.MinAbsPartialCharge(mol)
            feats['max_partial_charge'] = Descriptors.MaxPartialCharge(mol)
            feats['min_partial_charge'] = Descriptors.MinPartialCharge(mol)
        except Exception:
            feats['max_abs_partial_charge'] = float('nan')
            feats['min_abs_partial_charge'] = float('nan')
            feats['max_partial_charge'] = feats['min_partial_charge'] = float('nan')

        # Superfície
        feats['labute_asa'] = rdMolDescriptors.CalcLabuteASA(mol)
        feats.update(ChemFeaturizer.compute_descriptor_family(mol, 'PEOE_VSA'))
        feats.update(ChemFeaturizer.compute_descriptor_family(mol, 'SlogP_VSA'))
        feats.update(ChemFeaturizer.compute_descriptor_family(mol, 'EState_VSA'))

        # Halogênios
        feats.update(ChemFeaturizer.count_halogens(mol))

        # Grupos funcionais-chave
        flags = {
            name: ChemFeaturizer.bool_feature(mol, patt)
            for name, patt in ChemFeaturizer.SMARTS.items()
        }
        feats.update(flags)

        return feats

    def featurize_pure(self, df, n_jobs=-1):
        mols = self.smiles_to_mol(df['MOL'])
        feats = Parallel(n_jobs=n_jobs)(delayed(self.get_features)(mol) for mol in mols)
        feat_df = pd.DataFrame(feats)
        feat_df['T'] = df['T'].values
        feat_df['logV'] = df['logV'].values
        return feat_df

    def featurize_mix_parallel(self, df, n_jobs=-1):
        mols1 = self.smiles_to_mol(df['MOL_1'])
        mols2 = self.smiles_to_mol(df['MOL_2'])
        feat1 = Parallel(n_jobs=n_jobs)(delayed(self.get_features)(m) for m in mols1)
        feat2 = Parallel(n_jobs=n_jobs)(delayed(self.get_features)(m) for m in mols2)
        feat1 = pd.DataFrame(feat1).add_prefix('mol1_')
        feat2 = pd.DataFrame(feat2).add_prefix('mol2_')

        feat12 = pd.concat([feat1, feat2], axis=1)
        feat21 = pd.concat([feat2, feat1], axis=1)

        feat12['frac'] = df['MolFrac_1'].values
        feat12['T'] = df['T'].values
        feat12['logV'] = df['logV'].values

        feat21['frac'] = 1 - df['MolFrac_1'].values
        feat21['T'] = df['T'].values
        feat21['logV'] = df['logV'].values

        return pd.concat([feat12, feat21], axis=0).reset_index(drop=True)
