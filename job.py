from pathlib import Path
from warnings import filterwarnings

# Silence some expected warnings
filterwarnings("ignore")

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Draw, rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Neural network specific libraries
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint


url = "https://cloud-new.gdb.tools/index.php/s/ZfZM7itQf3rm6Sw/download"
df = pd.read_csv(url, index_col=0)
df = df.reset_index(drop=True)


# Converting IC50 values to pIC50 values
IC50_column = df['standard_value']
IC50_molar_column = IC50_column * 1e-9
pIC50_column = -np.log10(IC50_molar_column + 1e-10)

# Adding column containing pIC50 values
df['pIC50_value'] = pIC50_column

# Keep necessary columns
chembl_df = df[["pIC50_value", "smiles"]]

# Convert smiles to RDKit mol object
def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if method == "morgan2":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(fpg.GetCountFingerprint(mol))
    if method == "morgan3":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        return np.array(fpg.GetCountFingerprint(mol))
    else:
        print(f"Warning: Wrong method specified: {method}." " Default will be used instead.")
        return np.array(MACCSkeys.GenMACCSKeys(mol))

#Convert all SMILES strings to MACCS fingerprint
chembl_df["fingerprints_df"] = chembl_df["smiles"].apply(smiles_to_fp)

# Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(
    chembl_df["fingerprints_df"], chembl_df[["pIC50_value"]], test_size=0.3, random_state=42
)

def neural_network_model(hidden1, hidden2):
    model = Sequential()
    # First hidden layer
    model.add(Dense(hidden1, activation="relu", name="layer1"))
    # Second hidden layer
    model.add(Dense(hidden2, activation="relu", name="layer2"))
    # Output layer
    model.add(Dense(1, activation="linear", name="layer3"))

    # Compile model
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse", "mae"])
    return model

nb_epoch = 50
layer1_size = 64
layer2_size = 32

model = neural_network_model(layer1_size, layer2_size)



filepath = "best_weights.weights.h5"
checkpoint = ModelCheckpoint(
    str(filepath),
    monitor="loss",
    verbose=0,
    save_best_only=True,
    mode="min",
    save_weights_only=True,
)
callbacks_list = [checkpoint]


model.fit(
    np.array(list((x_train))).astype(float),
    y_train.values,
    epochs=nb_epoch,
    batch_size=16,
    callbacks=callbacks_list,
    verbose=0,
)
