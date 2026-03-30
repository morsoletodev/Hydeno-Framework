# DataSUS Machine Learning Pipeline

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
- `jupyter`: Jupyter server used to execute ml notebooks.

The flag `--no-deps` can be used to execute `process`, `train_model` and `linkage` services solo, since they are dependent on the previous (e.g. `process` depends on `acquire`).

## License
[GNU](./LICENSE)