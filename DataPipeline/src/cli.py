import typer

from .dataset import get_dataset
from .features import process_data
from .linkage import train_splink, predict_splink

app = typer.Typer()


@app.command()
def acquire():
    get_dataset()


@app.command()
def process():
    process_data()


@app.command()
def train():
    train_splink()


@app.command()
def linkage():
    predict_splink()
