# Data Pipeline

## Dependencies  
Dependencies can be found at [pyproject.toml](./pyproject.toml).

## Configuration  
Pipeline can be configured by editing `config.py` (general), `features.py` (preprocessing steps) and `linkage\train.py` (linkage model created).

## Code structure  
````
src/
 - linkage/         
    - predict.py # Uses a saved model to perform linkage (*)
    - train.py   # Builds a splink model and saves in models/

 - __main__.py   # Main file of the project
 - cli.py        # Command Line Interface definition
 - config.py     # Various configs used by other files
 - dataset.py    # Extracts the necessary databases
 - features.py   # Pre-process each database
````
`*` Can also perform build a model if none is found.
