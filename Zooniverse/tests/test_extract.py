# Test extract bounding classification for DeepForest
import os
import sys
import pytest
from .. import extract
from .. import aggregate

CLASSIFICATIONS_CSV = "data/everglades-watch-classifications.csv"
IMAGE_DATA = "data/everglades-watch-subjects.csv"


@pytest.fixture()
def run_aggregate():
    aggregate.run(CLASSIFICATIONS_CSV, min_version=272.359, download=False, generate=False,
                  savedir="output", debug=True)


def test_run(run_aggregate, tmpdir):
    # create an output image folder is needed
    if not os.path.exists("output/images/"):
        os.mkdir("output/images/")
    classification_out = "output/everglades-watch-classifications.shp"
    extract.run(image_data=IMAGE_DATA, classification_shp=classification_out, savedir=tmpdir)


def test_extract_empty(run_aggregate, tmpdir):
    parsed_annotations = "output/parsed_annotations.csv"
    extract.extract_empty(parsed_annotations, image_data=IMAGE_DATA, save_dir=tmpdir)
