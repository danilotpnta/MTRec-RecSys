# MTRec-RecSys

Repository for the course in Recommended Systems at University of Amsterdam 2024

# Getting Started
We recommend [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html#conda-environment) for environment management, and [VS Code](https://code.visualstudio.com/) for development. To install the necessary packages and run the example notebook:

```
# 1. Clone this repo using command line:
git clone https://github.com/danilotpnta/MTRec-RecSys.git
cd MTRec-RecSys/

```
Make sure you `deactivate` any other environemnt i.e. anaconda. For compatibility, we recommend to use `python 3.11`

Now, let's create a virtual environment (`venv`) and install the dependencies specified in the `pyproject.toml` file.

```
# 2. Create venv environment:
python3 -m venv venv
source  venv/bin/activate

# 3. Install the core ebrec package to the enviroment:
pip install .
```

## Running GPU
```
tensorflow-gpu; sys_platform == 'linux'
tensorflow-macos; sys_platform == 'darwin'
```

# Algorithms
To get started quickly, we have implemented a couple of News Recommender Systems, specifically, 
[Neural Recommendation with Long- and Short-term User Representations](https://aclanthology.org/P19-1033/) (LSTUR),
[Neural Recommendation with Personalized Attention](https://arxiv.org/abs/1907.05559) (NPA),
[Neural Recommendation with Attentive Multi-View Learning](https://arxiv.org/abs/1907.05576) (NAML), and
[Neural Recommendation with Multi-Head Self-Attention](https://aclanthology.org/D19-1671/) (NRMS). 
The source code originates from the brilliant RS repository, [recommenders](https://github.com/recommenders-team/recommenders). We have simply stripped it of all non-model-related code.


# Notebooks
To help you get started, we have created a few notebooks. These are somewhat simple and designed to get you started. We do plan to have more at a later stage, such as reproducible model trainings.
The notebooks were made on macOS, and you might need to perform small modifications to have them running on your system.

## Model training
We have created a [notebook](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/nrms_ebnerd.ipynb) where we train NRMS on EB-NeRD - this is a very simple version using the demo dataset.

## Data manipulation and enrichment
In the [dataset_ebnerd](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/dataset_ebnerd.ipynb) demo, we show how one can join histories and create binary labels.
