# Ensemble Models

## Notebooks
The following notebooks present an exploratory analysis of each ensembler with each sampler model chosen.
 - `1-catBoost.ipynb` 
 - `1-lightGBM.ipynb`
 - `1-randomForest.ipynb`
 - `1-xgboost.ipynb`

Lastly, `2-HPO.ipynb` performs hyperparameter optimization on the two best models, selected based on highest Recall and F1-score achieved.

## Dependencies  
Dependencies can be found at [pyproject.toml](./pyproject.toml).

## Configuration  
Pipeline can be configured by editing `config.py`.

## Code structure
````
src/
 - algorithms
    - ensemble.py # Ensemble models factory
    - sampler.py  # Sampler methods factory
 - config.py      # Various configs used by other files
 - dataset.py     # Load dataset to memory
 - pipeline.py    # Trainining and evaluation of models
 - report.py      # Evaluation methods
````
