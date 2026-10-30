# Environment setup
conda create --name drug_discovery_cuda --clone llm_scratch_cuda
conda activate drug_discovery_cuda
pip install rdkit
pip install scikit-learn seaborn


Important:
Don't do  $ conda update --all -c conda-forge -y because this will install a PyTorch version without CUDA 


# Verification
To check that CUDA, PyTorch and RDKit are working, run 01/version.py

# Notes
Build AI Drug Discovery Pipelines: Use AI models to discover new pharmaceutical drugs.

Chemical informatics, also known as chemoinformatics or cheminformatics
Cheminformatics sits at the intersection of chemistry, computer science, and data science. 
It focuses on storing, analyzing, searching, and manipulating chemical structure data using computers.
In the context of the Build AI Drug Discovery Pipelines book, cheminformatics represents the first step: data engineering step. 
It is the practice of taking raw chemical representations, like the SMILES strings and SDF datasets you and computing quantitative properties or molecular fingerprints, like for example the Lipinski Rule of Five descriptors. 
This mathematical translation is required, because AI and ML models can't interpret raw chemical drawings. They require the structured numerical arrays that cheminformatics libraries like RDKit can generate. They can use these to discover new pharmaceutical drugs.


Life Sciences: Scientific discipline dealing with human, animal, or plant life. Includes biotechnology, genomics, medical devices, agriculture and botany
The goal of life sciences is to understand biological processes and apply that knowledge
Pharma: Highly regulated industry dedicated to discovering, testing, producing, and marketing medications
Pharma relies heavily on chemistry, biologics and genomics to create treatments and vaccines for human or animal diseases.
Every pharmaceutical company is a life sciences company, but not the other way around


Non-NLP task: A task that doesn't require Natural Language Processing and can be solved by traditional ML and DL neural networks


