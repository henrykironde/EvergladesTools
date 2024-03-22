import os
import shlex
import sys
import subprocess
import tools

import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def create_mbtile(path, year, site, force_upload=False):
    basename = os.path.splitext(os.path.basename(path))[0]
    flight = basename.replace("_projected", "")

    if tools.get_event(basename) == "primary":
        flight = "_".join(basename.split('_')[0:4])

    working_dir = tools.get_working_dir()
    mbtiles_dir = os.path.join(working_dir, "mapbox", year, site)
    if not os.path.exists(mbtiles_dir):
        os.makedirs(mbtiles_dir)

    mbtiles_filename = os.path.join(mbtiles_dir, f"{flight}.mbtiles")
    if os.path.exists(mbtiles_filename):
        os.remove(mbtiles_filename)

    print("Creating mbtiles file")
    rio_command = f"rio mbtiles {path} -o {mbtiles_filename} --zoom-levels 17..24 -j 4 -f PNG"

    rio_command_list = shlex.split(rio_command)
    return_code = subprocess.call(rio_command_list)

    # Check the return code to see if the command was successful
    if return_code != 0:
        print(f"Rio command failed with return code {return_code}")
        return None

    if not os.path.exists(mbtiles_filename):
        print(f"{mbtiles_filename} was not created")
        return None

    return mbtiles_filename


if __name__ == "__main__":
    path = sys.argv[1]
    split_path = os.path.normpath(path).split(os.path.sep)
    year, site = split_path[6], split_path[7]
    force_upload = True

    # Create mbtiles
    file_path = create_mbtile(path, year, site, force_upload=force_upload)
