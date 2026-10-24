import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit.Chem import PandasTools
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# 1. Load Data
print("Loading FDA approved drugs dataset...")
fda_approved_df = pd.read_csv('data/fda_approved_drugs.csv', sep=',')
print(f"Total rows loaded: {len(fda_approved_df)}")

# 2. RDKit Molecule Parsing & Filtering
PandasTools.AddMoleculeColumnToFrame(
    fda_approved_df, smilesCol='Smiles', molCol='ROMol'
)
fda_approved_df = fda_approved_df[~fda_approved_df['ROMol'].isnull()]
print(f"Valid molecules parsed: {len(fda_approved_df)}")

# 3. Compute ECFP6 Fingerprints
print("Generating ECFP6 Morgan fingerprints (radius=3, size=1024)...")
morgan_gen = GetMorganGenerator(radius=3, fpSize=1024)
fda_approved_df['ECFP6'] = fda_approved_df['ROMol'].apply(
    lambda mol: morgan_gen.GetFingerprint(mol)
)

# 4. Dimensionality Reduction with PCA
print("Fitting PCA (n_components=4)...")
X = np.array([x for x in fda_approved_df['ECFP6']])
pca = PCA(n_components=4, random_state=42)
pca_drugs = pca.fit_transform(X)

fda_approved_df['PC1'] = pca_drugs.T[0]
fda_approved_df['PC2'] = pca_drugs.T[1]
fda_approved_df['PC3'] = pca_drugs.T[2]
fda_approved_df['PC4'] = pca_drugs.T[3]

print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

# 5. Visualizing Pairs (Display/Save)
print("Generating Seaborn pairplot...")
pairplot_fig = sns.pairplot(
    fda_approved_df,
    hue='USAN Stem',
    vars=['PC1', 'PC2', 'PC3', 'PC4'],
    palette='viridis'
)
pairplot_fig.savefig("pca_pairplot.png")
print("Saved pairplot to 'pca_pairplot.png'")

# 6. Binary Classification Targets
fda_approved_df['cillin'] = (
    fda_approved_df['USAN Stem'] == '-cillin'
).astype(int)
fda_approved_df['olol'] = (
    fda_approved_df['USAN Stem'] == '-olol'
).astype(int)

cillin_count = fda_approved_df['cillin'].sum()
olol_count = fda_approved_df['olol'].sum()
print(f"Target distribution -> '-cillin': {cillin_count}, '-olol': {olol_count}")

# 7. Logistic Regression on PC1 and PC3
print("Training Logistic Regression on [PC1, PC3] for '-cillin'...")
feature_cols = ['PC1', 'PC3']
lr_model_cillin = LogisticRegression()
lr_model_cillin.fit(fda_approved_df[feature_cols], fda_approved_df.cillin)

# 8. Decision Boundary Grid
xx, yy = np.mgrid[
    fda_approved_df.PC1.min():fda_approved_df.PC1.max():0.01,
    fda_approved_df.PC3.min():fda_approved_df.PC3.max():0.01
]

# Pass a DataFrame with matching column names to silence the scikit-learn warning
grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_cols)
probs = lr_model_cillin.predict_proba(grid_df)[:, 1].reshape(xx.shape)

print(f"Decision grid evaluated over shape {probs.shape}.")
print(f"Min probability: {probs.min():.4f}, Max probability: {probs.max():.4f}")
print("Pipeline run completed successfully.")
