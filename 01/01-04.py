# Processes the chemical structures of FDA-approved drugs into molecular ECFP6 fingerprints
# Applies PCA (Principal Component Analysis) to reduce the dimensionality of the fingerprints (from 1024D to 4D)
# Trains a logistic regression model to classify specific drug categories based on their principal components
# Scikit-learn is the right library as it natively supports Logistic Regression. Using Pytorch or Keras would be overkill as they require to implement id from scratch

# 1. Define (instantiate) a logistic regression object
# 2. Train the model using the PCA-reduced molecules (data) and the know categories (labels)

# Next steps, not in this script:
# 3. Test the trained model on a subset of molecules to verify its accuracy
# 4. Inference. Take a molecule, calculate the fingerprint, apply the PCA, pass it to the model. The model predicts to which category it belongs

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console
from rdkit.Chem import PandasTools
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Pairplot (grid of pair plots). Visualization that shows the relationships between the variables in a dataset in a grid.
FDA_APPROVED_PATH = "data/fda_approved_drugs.csv"
PCA_PAIRPLOT_PATH = "data/pca_pairplot.png"

# *main*
console = Console()

# 1. Load Data
console.print("\nLoading FDA approved drugs dataset.", style="gold1")
fda_approved_df = pd.read_csv(FDA_APPROVED_PATH, sep=',')
print(f"Total rows loaded: {len(fda_approved_df)}")

# 2. RDKit Molecule Parsing & Filtering
console.print("\nFingerprinting.", style="gold1")
PandasTools.AddMoleculeColumnToFrame(fda_approved_df, smilesCol='Smiles', molCol='ROMol')
fda_approved_df = fda_approved_df[~fda_approved_df['ROMol'].isnull()]
print(f"Valid molecules parsed: {len(fda_approved_df)}")

# 3. Compute ECFP6 Fingerprints
print("Generating ECFP6 Morgan fingerprints (radius=3, size=1024)...")
morgan_gen = GetMorganGenerator(radius=3, fpSize=1024)
fda_approved_df['ECFP6'] = fda_approved_df['ROMol'].apply(
    lambda mol: morgan_gen.GetFingerprint(mol)
)

# 4. Dimensionality Reduction with PCA
console.print("\nReducing number of dimensions", style="gold1")
print("Fitting PCA (n_components=4)...")
X = np.array([x for x in fda_approved_df['ECFP6']])
pca = PCA(n_components=4, random_state=42)
pca_drugs = pca.fit_transform(X)

fda_approved_df['PC1'] = pca_drugs.T[0]
fda_approved_df['PC2'] = pca_drugs.T[1]
fda_approved_df['PC3'] = pca_drugs.T[2]
fda_approved_df['PC4'] = pca_drugs.T[3]

print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

# 5. Generate the pairplot using Seaborn sns.pairplot()
console.print("\nGenerating Seaborn pairplot.", style="gold1")
pairplot_fig = sns.pairplot(
    fda_approved_df,
    hue='USAN Stem',
    vars=['PC1', 'PC2', 'PC3', 'PC4'],
    palette='viridis'
)
pairplot_fig.savefig(PCA_PAIRPLOT_PATH)
print(f"Pairplot saved to {PCA_PAIRPLOT_PATH}")

# 6. Binary Classification Targets
fda_approved_df['cillin'] = (fda_approved_df['USAN Stem'] == '-cillin').astype(int)
fda_approved_df['olol'] = (fda_approved_df['USAN Stem'] == '-olol').astype(int)

cillin_count = fda_approved_df['cillin'].sum()
olol_count = fda_approved_df['olol'].sum()
print(f"Target distribution -> '-cillin': {cillin_count}, '-olol': {olol_count}")

# 7. Logistic Regression on PC1 and PC3
console.print("\nDefine Logistic Regression model on [PC1, PC3] for '-cillin'.", style="gold1", highlight=False)
feature_cols = ['PC1', 'PC3']
lr_model_cillin = LogisticRegression()


console.print("\nTraining.", style="gold1")
lr_model_cillin.fit(fda_approved_df[feature_cols], fda_approved_df.cillin)

# 8. Decision Boundary Grid
xx, yy = np.mgrid[fda_approved_df.PC1.min():fda_approved_df.PC1.max():0.01,fda_approved_df.PC3.min():fda_approved_df.PC3.max():0.01]

# Pass a DataFrame with matching column names to silence the scikit-learn warning
grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_cols)
probs = lr_model_cillin.predict_proba(grid_df)[:, 1].reshape(xx.shape)

print(f"Decision grid evaluated over shape {probs.shape}.")
print(f"Min probability: {probs.min():.4f}, Max probability: {probs.max():.4f}")
print("Pipeline run completed successfully.\n")

# Two adaptations are possible for this code, as an academic exercise. For production, SCIKit-Learn is the right choice.
# 1. Move from Scikit-learn to Pytorch
# 2. Move from Pytorch to Pytorch-CUDA, initializing a device, moving the objects to GPU-VRAM and doing the calculations there