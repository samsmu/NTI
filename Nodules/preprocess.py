import numpy as np
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
from skimage.transform import resize

def create_lung_mask(img):

    #Standardize the pixel values
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std

    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean

    # Using Kmeans to separate foreground (radio-opaque tissue)
    # and background (radio transparent tissue ie lungs)
    # Doing this only on the center of the image to avoid 
    # the non-tissue parts of the image as much as possible
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    #
    # I found an initial erosion helful for removing graininess from some of the regions
    # and then large dialation is used to make the lung region 
    # engulf the vessels and incursions into the lung cavity by 
    # radio opaque tissue
    #
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    #
    #  Label each region and obtain the region properties
    #  The background region is removed by removing regions 
    #  with a bbox that is to large in either dimnsion
    #  Also, the lungs are generally far away from the top 
    #  and bottom of the image, so any regions that are too
    #  close to the top and bottom are removed
    #  This does not produce a perfect segmentation of the lungs
    #  from the image, but it is surprisingly good considering its
    #  simplicity. 
    #
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    
    return mask

def apply_lung_mask(img, mask, node_mask = None):
    new_size = [512,512]   # we're scaling back up to the original size of the image
    img= mask*img          # apply lung mask
    #
    # renormalizing the masked image (in the mask region)
    #
    new_mean = np.mean(img[mask>0])  
    new_std = np.std(img[mask>0])
    #
    #  Pulling the background color up to the lower end
    #  of the pixel range for the lungs
    #
    old_min = np.min(img)       # background color
    img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
    img = img-new_mean
    img = img/new_std
    #make image bounding box  (min row, min col, max row, max col)
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    #
    # Finding the global min and max row over all regions
    #
    min_row = 512
    max_row = 0
    min_col = 512
    max_col = 0
    for prop in regions:
        B = prop.bbox
        if min_row > B[0]:
            min_row = B[0]
        if min_col > B[1]:
            min_col = B[1]
        if max_row < B[2]:
            max_row = B[2]
        if max_col < B[3]:
            max_col = B[3]
    width = max_col-min_col
    height = max_row - min_row
    if width > height:
        max_row=min_row+width
    else:
        max_col = min_col+height
    # 
    # cropping the image down to the bounding box for all regions
    # (there's probably an skimage command that can do this in one line)
    # 
    img = img[min_row:max_row,min_col:max_col]
    mask =  mask[min_row:max_row,min_col:max_col]
    if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
        new_img = None
        new_node_mask = None
    else:
        # moving range to -1 to 1 to accomodate the resize function
        mean = np.mean(img)
        img = img - mean
        min = np.min(img)
        max = np.max(img)
        img = img/(max-min)
        new_img = resize(img,[512,512])
        if node_mask is None:
            new_node_mask = None
        else:
            new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col],[512,512])
    return (new_img, new_node_mask)