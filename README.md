# DataSUS Machine Learning Pipeline

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
[![Black](https://img.shields.io/badge/code%20style-black-orange.svg)](https://github.com/psf/black)
[![Uses the Cookiecutter Data Science project template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Installation & Usage

The command `docker compose` is responsible for building and execution the pipeline.

```bash
docker compose up <command> <--no-deps>
```

The available commands are listed below:
- `acquire`: Download datasets, and merge into single file.
- `process`: Adjust format and schema of acquired files.
- `train_model`: Creates a splink model.
- `linkage`: Performs the linkage between datasets;
- `jupyter`: Jupyter server used to execute ml notebooks. Check [EnsembleModels](EnsembleModels/README.md) for more information.

The flag `--no-deps` can be used to execute `process`, `train_model` and `linkage` services solo, since they are dependent on the previous (e.g. `process` depends on `acquire`).

## License
[GNU](./LICENSE)