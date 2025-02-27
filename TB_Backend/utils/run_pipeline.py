
from pathlib import Path
import logging

from utils.loader import load
from utils.get_map import get_map
from utils.smear_pipeline import smear_pipeline

logger = logging.getLogger(__name__)


def run_pipeline(czi_path: Path) -> Path:
    """
    Run the smear pipeline on a given czi file.
    Parameters
    ----------
    czi_path : Path
        Path to the czi file to process.

    Returns
    -------
    Path
        Path to the original image.

    """

    img, loader = load(czi_path)
    logger.info(f"Loaded image from {czi_path}")

    logger.info(f"Processing image")
    bacilli_information = smear_pipeline(img, loader)
    logger.info(f"Processing completed")

    logger.info(f"Creating Heatmap")
    og_path, map_path = get_map(loader.reader, bacilli_information[1])
    logger.info(f"Heatmap created")

    return og_path, map_path
