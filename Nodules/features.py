import numpy as np

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

def get_region_from_map(slice_npy):
    thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)
    label_image = label(thr)
    labels = label_image.astype(int)
    regions = regionprops(labels)
    return regions

def get_features_from_region(region):

    minr, minc, maxr, maxc = region.bbox
    nodule_features =[region.area,
                       region.eccentricity,
                       region.equivalent_diameter,
                       region.centroid[0]*region.area,
                       region.centroid[1]*region.area,
                       minr,
                       minc,
                       maxr,
                       maxc]
    return nodule_features