import os
import skimage
import openslide
import numpy as np
from tifffile import TiffWriter
from matplotlib import pyplot as plt
from pybioimageutils import visualize
import slides

# image_A_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259270.svs"

# image_B_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261082.svs"

image_A_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261081.svs"

image_B_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259267.svs"

output_ds_level = 2
output_filename = 'merged_261081_259267_level2'

# ---Begin processing---
# Create slide objects
slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

# shift = slides.calculate_shift(slide_B, slide_A, ds_level=1)
shift = (600, -2788)
num_subifds = slide_A.level_count - 1

output_size = slide_A.level_dimensions[output_ds_level]

sz_tile = 128

pixelsize = float(slide_A.properties[openslide.PROPERTY_NAME_MPP_X])

options = dict(
    tile=(sz_tile, sz_tile),
    compression='jpeg',
    dtype=np.uint8,
    photometric='rgb',
    resolutionunit='CENTIMETER',
)

# For single image
with TiffWriter(('./export/' + output_filename + '.tif'), bigtiff=True) as tif:
    metadata = {
            'PhysicalSizeX': pixelsize,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pixelsize,
            'PhysicalSizeYUnit': 'µm',
    }
        
    tif.write(
        slides.register_tiled_image(slide_B, slide_A, shift, ds_level=output_ds_level, tile_size=sz_tile),
        shape=(output_size[1], output_size[0], 3),
        resolution=(1e4 / pixelsize, 1e4 / pixelsize),
        metadata=metadata,
        **options,
        )
    



# with TiffWriter('./export/merged_259270_261082.tif', bigtiff=True) as tif:
#     metadata = {
#             'PhysicalSizeX': pixelsize,
#             'PhysicalSizeXUnit': 'µm',
#             'PhysicalSizeY': pixelsize,
#             'PhysicalSizeYUnit': 'µm',
#             'MapAnnotation': {  # for OMERO
#                 'Namespace': 'openmicroscopy.org/PyramidResolution',
#                 '1': '256 256',
#                 '2': '128 128',
#              },
#     }
        
#     tif.write(
#         slides.register_tiled_image(slide_B, slide_A, shift, ds_level=0, tile_size=sz_tile),
#         shape=(slide_A.dimensions[1], slide_A.dimensions[0], 3),
#         subifds=num_subifds,
#         resolution=(1e4 / pixelsize, 1e4 / pixelsize),
#         metadata=metadata,
#         **options,
#         )
    
#     for i in range(num_subifds):
#         subimage_dimension = slide_A.level_dimensions[i + 1]
#         mag = slide_A.level_downsamples[i + 1]

#         tif.write(
#                 slides.register_tiled_image(slide_B, slide_A, shift, ds_level=(i + 1), tile_size=sz_tile),
#                 shape=(subimage_dimension[1], subimage_dimension[0], 3),
#                 subfiletype=1,
#                 metadata=metadata,
#                 resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
#                 **options,
#         )
