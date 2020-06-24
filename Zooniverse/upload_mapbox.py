import os
import subprocess
import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling

subprocess.call("/orange/ewhite/everglades/mapbox/source_token.txt", shell =True)

#Files to upload to mapbox
files_to_upload = ['/orange/ewhite/everglades/WadingBirds2020/CypressCity/CypressCity_03_25_2020.tif',
'/orange/ewhite/everglades/WadingBirds2020/Jerrod/Jerrod_03_24_2020.tif',
'/orange/ewhite/everglades/WadingBirds2020/Jetport/JetportSouth_03_23_2020.tif',
'/orange/ewhite/everglades/WadingBirds2020/Joule/Joule_03_24_2020.tif',
'/orange/ewhite/everglades/WadingBirds2020/StartMel/StartMel_03_24_2020.tif',
"/orange/ewhite/everglades/WadingBirds2020/Vacation/Vacation_03_24_2020.tif"]

dst_crs = rio.crs.CRS.from_epsg("3857")

for path in files_to_upload:
     src = rio.open(path)
     
     with rio.open(path) as src:
          transform, width, height = calculate_default_transform(
              src.crs, dst_crs, src.width, src.height, *src.bounds)
          kwargs = src.meta.copy()
          kwargs.update({
              'crs': dst_crs,
             'transform': transform,
             'width': width,
             'height': height
         })
     
          #create output filename
          out_filename = "{}_projected.tif".format(os.path.splitext(path)[0])
          
          with rio.open(out_filename, 'w', **kwargs) as dst:
               for i in range(1, src.count + 1):
                    reproject(
                     source=rio.band(src, i),
                     destination=rio.band(dst, i),
                     src_transform=src.transform,
                     src_crs=src.crs,
                     dst_transform=transform,
                     dst_crs=dst_crs,
                     resampling=Resampling.nearest)
                        
    ##Project to web mercator
    #create output filename
     basename = os.path.splitext(os.path.basename(path))[0]
     mbtiles_filename = "/orange/ewhite/everglades/mapbox/{}.mbtiles".format(basename)
     subprocess.call("rio mbtiles {} -o {} --zoom-levels 17..22 -j 4 -f PNG --overwrite".format(out_filename, mbtiles_filename), shell=True)
    
     ##Generate tiles
     subprocess.call("mapbox upload bweinstein.{} {}".format(basename,mbtiles_filename), shell=True)

