# Test deepforest development
import glob
import os

import geopandas as gp
import pandas as pd
import pytest
import rasterio
from pytorch_lightning.loggers import CometLogger

from .. import aggregate
from .. import extract
from ..species_model import create_species_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
API_KEY = "ypQZhYfs3nSyKzOfz13iuJpj2"
PROJECT_NAME = "everglades-species"
WORKSPACE = "bw4sz"
CLASSIFICATIONS_CSV = "data/everglades-watch-classifications.csv"
IMAGE_DATA = "data/everglades-watch-subjects.csv"


# Setup method
@pytest.fixture()
def extract_images(tmpdir):
    aggregate.run(CLASSIFICATIONS_CSV, min_version=300, download=False, generate=False, savedir="output", debug=True)
    classification_out = "output/everglades-watch-classifications.shp"
    extract.run(image_data=IMAGE_DATA, classification_shp=classification_out, savedir=tmpdir)


@pytest.fixture()
def annotations(extract_images, tmpdir):
    annotations = create_species_model.format_shapefiles(shp_dir=tmpdir)
    return annotations


def test_shapefile_to_annotations(extract_images, tmpdir):
    rgb_path = glob.glob("{}/*.png".format(tmpdir))[0]
    shp = "{}/{}.shp".format(tmpdir, os.path.splitext(os.path.basename(rgb_path))[0])
    df = create_species_model.shapefile_to_annotations(shapefile=shp, rgb_path=rgb_path)
    assert all(df.columns == ["image_path", "xmin", "ymin", "xmax", "ymax", "label"])

    # assert that the coordinates are in the image system
    with rasterio.open(rgb_path) as src:
        height, width = src.shape

    assert (df.iloc[0].xmin >= 0) & (df.iloc[0].xmax <= width)
    assert (df.iloc[0].ymin >= 0) & (df.iloc[0].ymax <= height)

    # Assert total number of records
    gdf = gp.read_file(shp)
    assert gdf.shape[0] == df.shape[0]

    # Assert no duplicates
    gdf_dropped_duplicates = gdf.drop_duplicates()
    assert gdf_dropped_duplicates.shape[0] == gdf.shape[0]


def test_empty_image():
    image = ["a", "a", "a", "b", "b"]
    scores = ["0.1", "0.1", "0.1", "0.2", "0.9"]
    precision_curve = pd.DataFrame({"image": image, "score": scores})
    empty_recall = create_species_model.empty_image(precision_curve, threshold=0.15)
    assert empty_recall == 0.5


def test_plot_recall_curve():
    image = ["a", "a", "a", "b", "b"]
    scores = ["0.1", "0.1", "0.1", "0.2", "0.9"]
    precision_curve = pd.DataFrame({"image": image, "score": scores})

    ax1 = create_species_model.plot_recall_curve(precision_curve)


def test_format_shapefiles(extract_images, tmpdir):
    results = create_species_model.format_shapefiles(shp_dir=tmpdir)
    assert all(results.columns == ["image_path", "xmin", "ymin", "xmax", "ymax", "label"])
    assert results.xmin.dtype == int

    # Assert no duplicates
    results_dropped_duplicates = results.drop_duplicates()
    assert results_dropped_duplicates.shape[0] == results.shape[0]


def test_split_test_train(extract_images, annotations):
    train, test = create_species_model.split_test_train(annotations)

    # Assert no overlapping cases and known deepforest format
    assert not test.empty
    assert not train.empty
    assert all(train.columns == ["image_path", "xmin", "ymin", "xmax", "ymax", "label"])
    assert all(test.columns == ["image_path", "xmin", "ymin", "xmax", "ymax", "label"])
    assert test[test.image_path.isin(train.image_path.unique())].empty

    # Assert that data is same total sum
    # assert annotations.shape[0] == (test.shape[0] + train.shape[0])

    # Assert no duplicates
    train_dropped_duplicates = train.drop_duplicates()


def test_train_model(extract_images, annotations, tmpdir):
    comet_logger = CometLogger(api_key=API_KEY,
                               project_name=PROJECT_NAME, workspace=WORKSPACE)

    train, test = create_species_model.split_test_train(annotations)
    train_path = os.path.join(tmpdir, "train.csv")
    train.to_csv(train_path, index=False)
    test_path = os.path.join(tmpdir, "test.csv")
    test.to_csv(test_path, index=False)
    create_species_model.train_model(train_path=train_path, test_path=test_path, epochs=1, comet_logger=comet_logger,
                                     debug=True)
