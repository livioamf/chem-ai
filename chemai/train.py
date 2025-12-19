import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def train_test_split(smiles1, smiles2=None, test_size=0.2, random_state=13, **features):

    smiles1 = list(smiles1)

    if smiles2 is not None:
        smiles2 = list(smiles2)
        groups = [f'{[mol1]} {[mol2]}' for mol1, mol2 in zip(smiles1, smiles2)]
        features = {'smiles_1': smiles1, 'smiles_2': smiles2, **features}
    else:
        groups = smiles1
        features = {'smiles': smiles1, **features}

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))

    train = {}
    test = {}

    for col, arr in features.items():
        if isinstance(arr, list):
            train[col] = [arr[i] for i in train_idx]
            test[col] = [arr[i] for i in test_idx]
        else:
            arr_np = np.array(arr)
            train[col] = arr_np[train_idx]
            test[col] = arr_np[test_idx]

    return train, test
