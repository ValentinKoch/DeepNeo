import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import time
import config
from scipy.ndimage.measurements import label
from scipy.ndimage import generate_binary_structure
from scipy.spatial import KDTree

def get_calc_score(prediction_df, mask_dir,pixel_spacing,slice_thickness):

    all_masks=sorted(Path(mask_dir).glob("*.tif"))
    score=0
    max_degree={"value":0, "index":-1}
    max_thickness={"value":0, "index":-1,"points":None,"im_path":None}
    max_length={"value":0, "start_index":-1,"end_index":-1}
    time_deg_tot=0
    time_thick_tot=0
    time_center_tot=0
    time_a_line_tot=0

    if(pixel_spacing == 0):
        pixel_spacing = config.DEFAULT_PIXEL_SPACING


    for i,mask in enumerate(all_masks):
        if prediction_df.loc[int(mask.stem)]["use_for_summary"]:

            open_mask=np.array(Image.open(mask))
            time0=time.time()
            center=get_center(open_mask)
            time_center_tot+=time.time()-time0

            time0=time.time()
            a_line,_ = get_a_line(open_mask,center)
            time_a_line_tot+=time.time()-time0

            time0=time.time()
            degree=calculate_max_degree(a_line)
            time_deg_tot+=time.time()-time0

            time0=time.time()
            thickness,pointA,pointB=get_max_thickness(open_mask,pixel_spacing)
            time_thick_tot+=time.time()-time0
            if max_degree["value"]<degree:
                max_degree["value"]=degree
                max_degree["index"]=i

            if max_thickness["value"]<thickness:
                max_thickness["value"]=thickness
                max_thickness["index"]=i
                max_thickness["points"]=(pointA,pointB)
                max_thickness["im_path"]=str(mask)

    time0=time.time()
    stacked=stack_masks(prediction_df,mask_dir)
    print("time to get stacked masks: " ,time.time()-time0)
    time0=time.time()

    _, max_length["value"], max_length["start_index"] , max_length["end_index"] = longest_connected_component(stacked,slice_thickness)
    
    print("time to get connected component: " ,time.time()-time0)
    print("time thickness:", time_thick_tot)
    print("time degree:", time_deg_tot)
    print("time center:", time_center_tot)
    print("time a_line:", time_a_line_tot)

    score+=max_thickness["value"]>=0.5
    score+=2*(max_degree["value"]>=180)
    score+=max_length["value"]>=5

    return max_thickness,max_degree,max_length, score #e.g .({'value': 0.7441800390818853, 'index': 17, 'points': (array([379, 176]), array([287, 234]))}, {'value': 104.765625, 'index': 140}, {'value': 6.399998816, 'start_index': 11, 'end_index': 43}, 2)

def load_mask(mask_path, calc_class, use_for_summary):
    # Load the mask and convert it to a binary array with the specified class
    mask = np.array(Image.open(mask_path))
    if use_for_summary:
        binary_mask = np.where(mask == calc_class, 1, 0)
    else:
        binary_mask= np.zeros_like(mask)
    return binary_mask

def stack_masks(prediction_df, mask_dir, calc_class=config.CALC_CLASS):
    # Load the paths of all mask files
    all_masks = sorted(Path(mask_dir).glob("*.tif"))

    # Create a thread pool to load the masks
    pool = ThreadPool()

    # Use the pool to asynchronously load the binary masks and preserve order
    results = []
    for mask_path in all_masks:

        use_for_summary= prediction_df.loc[int(mask_path.stem)]["use_for_summary"]
        result = pool.apply_async(load_mask, args=(mask_path, calc_class,use_for_summary))
        results.append(result)
    results = [result.get() for result in results]

    # Stack the binary masks along a new 3rd dimension
    stacked = np.stack(results, axis=2)

    return stacked


def longest_connected_component(arr, slice_thickness):
    # Find connected regions
    labeled_array, num_features = label(arr)

    # Find the size of each labeled region along the 3rd dimension

    layer_list = [np.unique(labeled_array[:, :, i]) for i in range(labeled_array.shape[2])]


        #region_list = [region for region_list in region_lists for region in region_list]
    region, count = np.unique(np.array(flatten(layer_list)), return_counts=True)

    # sort count in descending order and get the second element
    try:
        second_largest_count = np.sort(count)[::-1][1]

        # find the index of the second largest count in the count array
        second_largest_index = np.where(count == second_largest_count)[0][0]

        # get the corresponding region from the region array
        second_largest_region = region[second_largest_index]

        all_region_indices = np.where(labeled_array == second_largest_region)
        min_index=np.min(all_region_indices[2])

    except IndexError:
        return 0, 0, None, None

    # as largest region is always background
    return second_largest_region, second_largest_count * slice_thickness, min_index, min_index + second_largest_count-1


def flatten(a):
    return [c for b in a for c in flatten(b)] if hasattr(a, '__iter__') else [a]

def calculate_max_degree(a_line, calc_class=config.CALC_CLASS):
    polar_mask_concat = np.concatenate((a_line, a_line), axis=0)
    consecutive_calc = polar_mask_concat[:, 0]

    # Remove consecutive duplicates of non-calc_class entries
    noncalc_indices = np.where(consecutive_calc != calc_class)[0]
    remove_indices = np.where(np.diff(noncalc_indices) == 1)[0]
    consecutive_calc = np.delete(consecutive_calc, noncalc_indices[remove_indices + 1])

    # Split the array at the remaining indices
    split_indices = np.where(consecutive_calc != calc_class)[0]
    subarrays = np.split(consecutive_calc, split_indices + 1)

    # Filter subarrays that do not contain calc_class
    filtered_arrays = [subarray for subarray in subarrays if np.any(subarray == calc_class)]

    if len(filtered_arrays) > 0:
        # Find the longest filtered subarray and its length
        longest_array = max(filtered_arrays, key=len)
        length = len(longest_array) - 1
        if length >= len(a_line):
            length -= len(a_line)
    else:
        length = 0

    # Calculate the maximum degree using NumPy operations
    degree = (length * 360) / len(a_line)

    return degree

def polarize_mask(img, p=None):
    """
    Transform the input image to its polar representation.
    :param img: The input image to transform.
    :param p: The center point for the polar transformation.
    :return: The polar representation of the input image.
    """
    img = np.array(img)
    if p is None:
        p = (img.shape[0]/2., img.shape[1]/2.)
    value = np.sqrt(((p[0]-img.shape[0])**2.0)+((p[1]-img.shape[1])**2.0))
    polar_image = cv2.linearPolar(img, p, value, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    return polar_image


def get_a_line(mask,center):
    polar_mask=polarize_mask(mask,center)
    arr_lesion=[int(1 in n) for n in polar_mask]
    arr_calc=[int(2 in n)*2 for n in polar_mask]
    arr_joined= np.max([arr_lesion,arr_calc],axis=0) 
    return np.transpose(np.repeat([arr_joined],1,axis=0)),polar_mask

def get_center(mask, lumen_index=config.LUMEN_CLASS_CALC):
    lumen = np.where(np.array(mask) == lumen_index)
    if (len(lumen[0])) > 0:
        center = [np.average(indices) for indices in lumen]
        center[0] = center[0] / len(mask[0])
        center[1] = center[1] / len(mask[1])
    else:
        center = [0.5, 0.5]
    #center = [0.4, 0.4]# first:up/down second left right
    return [center[1]*mask.shape[0], center[0]*mask.shape[1]]



def get_connected_regions(binary_array):
    """
    Takes a binary 2D numpy array as input and returns all regions that are connected by one, 
    each as a binary mask. Diagonally connected regions are considered to be one region.
    
    Args:
        binary_array: A binary 2D numpy array.
    
    Returns:
        A list of binary masks, each representing a connected region.
    """
    
    # Define a binary structure for 8-connectedness (including diagonals)
    structure = generate_binary_structure(2, 2)
    
    # Find connected components in the binary array using 8-connectedness
    labeled_array, num_features = label(binary_array, structure=structure)
    
    # Create a list to store the binary masks of each region
    regions = []
    
    # Loop over each labeled region
    for i in range(1, num_features+1):
        
        # Create a binary mask for the current region
        region_mask = np.zeros_like(binary_array)
        region_mask[labeled_array == i] = 1
        
        # Add the mask to the list of regions
        regions.append(region_mask)
    
    # Return the list of regions
    return regions  


def get_max_thickness(mask,pixel_spacing,calc_class=config.CALC_CLASS,lumen_class=config.LUMEN_CLASS_CALC):
    max_dist = 0
    max_dist_p0 = None
    max_dist_p2 = None
    lumen_binary = set_surrounded_to_zero(np.where(mask==lumen_class,1,0))
    calc_binary_temp = set_surrounded_to_zero(np.where(mask==calc_class,1,0))
    calc_region_binaries=get_connected_regions(calc_binary_temp)
    idx_lumen = np.argwhere(lumen_binary == 1)
    idx_tree_lumen = KDTree(idx_lumen)

    for calc_region_binary in calc_region_binaries:
    # Find the indices of all points in arr1 with value 1
        idx_calc = np.argwhere(calc_region_binary == 1)
        idx_tree_calc = KDTree(idx_calc)
        

        for i in range(len(idx_calc)):
            # Find the nearest lumen pixel to the current calc pixel
            _, idx_saved = idx_tree_lumen.query(idx_calc[i])
            p1 = idx_lumen[idx_saved]
            
            # Find the index of the closest calc point to the saved lumen point
            _, idx_closest = idx_tree_calc.query(p1)
            p2 = idx_calc[idx_closest]
            
            # Calculate the distance between p2 and p0
            dist = np.linalg.norm(p2 - idx_calc[i])
            
            # Save the maximum distance and corresponding points p0 and p2
            if dist > max_dist:
                max_dist = dist
                max_dist_p0 = idx_calc[i]
                max_dist_p2 = p2

    
    length_in_mm=max_dist*pixel_spacing*1024/mask.shape[0]
    return length_in_mm, max_dist_p0, max_dist_p2


def set_surrounded_to_zero(arr):
    # Check if input array is 2D
    
    # Create a boolean mask of elements that are ones surrounded by ones
    mask = (arr == 1) & \
           (np.roll(arr, 1, axis=0) == 1) & \
           (np.roll(arr, -1, axis=0) == 1) & \
           (np.roll(arr, 1, axis=1) == 1) & \
           (np.roll(arr, -1, axis=1) == 1)
    
    # Initialize an array of the same shape as the input array
    result = np.copy(arr)
    
    # Set the elements of the result array that match the mask to zero
    result[mask] = 0
    
    return result


if __name__ == "__main__":
    time0=time.time()
    f=get_calc_score("/Users/valentin.koch/Downloads/Case #1_2023-01-24-11-01/masks", config.DEFAULT_PIXEL_SPACING, config.DEFAULT_SLICE_THICKNESS)
    print(f)
    print("time passed: ", time.time()-time0)