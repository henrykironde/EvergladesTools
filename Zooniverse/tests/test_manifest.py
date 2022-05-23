# Test zooniverse upload and data post-processing
import os

import pytest
import rasterio

from .. import manifest
from .. import tile_raster


@pytest.fixture()
def PATH():
    return "data/Vacation_03192020_203.tif"


@pytest.fixture()
def projected_path(PATH):
    projected_path = manifest.utm_project(PATH)
    return projected_path


def test_utm_project(PATH):
    projected_path = manifest.utm_project(PATH)
    raster = rasterio.open(projected_path)
    str(raster.crs) == 'EPSG:32617'


def test_is_white():
    assert not manifest.is_white("data/Vacation_03192020_203.tif")


def test_tile_raster(projected_path):
    saved_file = tile_raster.run(path=projected_path, save_dir="output/", patch_size=100)
    basename = os.path.splitext(os.path.basename(projected_path))[0]
    assert os.path.exists("output/{}_0.tif".format(basename))
