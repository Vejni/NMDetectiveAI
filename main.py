import typer

from NMD.modeling.train import app as train_app
from NMD.modeling.predict import app as predict_app
from NMD.modeling.sweep import app as sweeper_app
from NMD.modeling.generate_genome_wide_bigwig import app as bigwig_app
from NMD.data.data import app as dataset_app
from NMD.data.DMS import app as dms_app
from NMD.plots import app as plots_app
from NMD.manuscript.manuscript_app import app as manuscript_app

app = typer.Typer(name="main", add_completion=False, help="This is a demo app.")
app.add_typer(dataset_app, name="dataset", help="Process the dataset.")
app.add_typer(dms_app, name="dms", help="Process DMS datasets.")
app.add_typer(train_app, name="train", help="Train the model.")
app.add_typer(predict_app, name="predict", help="Make predictions with the model.")
app.add_typer(bigwig_app, name="bigwig", help="Generate genome-wide bigWig track.")
app.add_typer(sweeper_app, name="sweep", help="Run hyperparameter sweeps.")
app.add_typer(plots_app, name="plots", help="Generate plots for analysis.")
app.add_typer(manuscript_app, name="manuscript", help="Generate manuscript figures.")

if __name__ == "__main__":
    app()
