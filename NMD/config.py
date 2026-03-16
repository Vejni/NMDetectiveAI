from pathlib import Path
from loguru import logger
from NMD.utils import set_seeds

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
LARGE_DATA_DIR = "/g/strcombio/fsupek_franklin/mveiner/Data/"
OUT_DIR = PROJ_ROOT / "out"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

MANUSCRIPT_FIGURES_DIR = PROJ_ROOT / "manuscript" / "figures"
MANUSCRIPT_SUPPLEMENTARY_FIGURES_DIR = PROJ_ROOT / "manuscript" / "supplementary" / "figures"
MANUSCRIPT_TABLES_DIR = PROJ_ROOT / "manuscript" / "supplementary" / "tables"

# Global random seed
SEED = 42
VAL_CHRS = ["chr1", "chr17"]
WANDB_PROJECT = "NMD"
GENCODE_VERSION = "gencode.v26"
MAX_TRANSCRIPT_LENGTH = 20000

# Colour schemes
EVADING_2_TRIGGERING_COLOUR_GRAD = ['#ff9e9d', '#ffb3b3', '#ffc8c8', '#ffdddd', '#022778']
CONTRASTING_2_COLOURS = ['#ff9e9d', '#022778']
CONTRASTING_3_COLOURS = ['#ff9e9d', "#2d8b4d", '#022778']
COLOURS = ['#fb731d', '#fcbb01', '#ff9e9d', '#ffdfcb', '#d6f3ff', '#2778ff', '#022778', '#2d8b4d', '#034e7b']

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

set_seeds(SEED)
