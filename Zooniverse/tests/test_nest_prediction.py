# Test nest_detection
import os

import geopandas
import pytest

from .. import nest_detection

is_travis = 'TRAVIS' in os.environ


def test_load_files():
    df = nest_detection.load_files("data/predictions/")
    assert not df.empty
    assert df.Site.unique() == ["Joule"]


def test_compare_site():
    df = nest_detection.load_files("data/predictions/")
    results = nest_detection.compare_site(df)
    assert not results.empty


def test_detect_nests():
    filename = nest_detection.detect_nests("data/predictions/", savedir="output/")
    assert os.path.exists(filename)


# Only run on Hipergator with access to the rgb data
@pytest.mark.skipif(is_travis, reason="Cannot load comet on TRAVIS")
def test_detect_nests():
    filename = nest_detection.detect_nests("data/predictions/", savedir="output/")
    rgb_pool = nest_detection.find_files()
    nest_detection.extract_nests(filename, rgb_pool=rgb_pool, savedir="output", upload=True)
    gdf = geopandas.read_file(filename)
    grouped = gdf.groupby("target_ind")
    for name, group in grouped:
        if group.shape[0] < 3:
            continue
        for index, row in group.iterrows():
            filename = "output/{}/{}_{}_{}.png".format(index, row["target_ind"], row["Site"], row["Date"])
            os.path.exists(filename)
