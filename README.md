# Environment setup
conda create --name drug_discovery_cuda --clone llm_scratch_cuda
conda activate drug_discovery_cuda
pip install rdkit
pip install scikit-learn seaborn


Important:
Don't do  $ conda update --all -c conda-forge -y because this will install a PyTorch version without CUDA 


# Verification
To check that CUDA, PyTorch and RDKit are working, run 01/version.py
