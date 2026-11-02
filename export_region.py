import os
import openslide
import slides
from tifffile import imwrite
import numpy as np

# imageA_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259270.svs"
# imageB_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261082.svs"

imageA_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259267.svs"
imageB_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261081.svs"

# Create slide objects
slideA = openslide.OpenSlide(imageA_path)
#slideB = openslide.OpenSlide(imageB_path)

topleft, region_size = slides.get_region(slideA)

img = slideA.read_region(topleft, 0, region_size)
img = np.array(img.convert('RGB'))

imwrite('/primary/instruments/confocal_core/Jian HPC/OIC-244/data/259267_261081/259267.tif', img)

# Create slide objects
slideB = openslide.OpenSlide(imageB_path)

topleft, region_size = slides.get_region(slideB)

img = slideB.read_region(topleft, 0, region_size)
img = np.array(img.convert('RGB'))

imwrite('/primary/instruments/confocal_core/Jian HPC/OIC-244/data/259267_261081/261081.tif', img)