import csv
import hashlib
import os
import random
import shutil
import string
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from time import localtime, strftime, time
from typing import List
import zipfile
from datetime import datetime


import boto3
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
from PIL import Image,UnidentifiedImageError
from scipy.ndimage import label

import config
from html_handler import create_summary_html
from pydicom.errors import InvalidDicomError


def get_random_string(length):
    # With combination of lower and upper case
    result_str = "".join(random.choice(string.ascii_letters) for i in range(length))
    return result_str


def convert_image(image):
    if image.mode == 'F':
        image = image.point(lambda x: x*255, 'F').convert('L')
    elif image.mode == 'I;16':
        image = image.point(lambda x: x/65535*255, 'I').convert('L')
    return image

def colour_np_array(arr):
    lut_desc = [256, 0, 16]
    # A value of 0 = 2^16 entries
    nr_entries = lut_desc[0] or 2**16

    # May be negative if Pixel Representation is 1
    first_map = lut_desc[1]
    # Actual bit depth may be larger (8 bit entries in 16 bits allocated)
    nominal_depth = lut_desc[2]
    dtype = np.dtype("uint{:.0f}".format(nominal_depth))

    luts = []

    # LUT Data is described by PS3.3, C.7.6.3.1.6
    r_lut = b'\x00\x00\x80\x80\x00\x00\x80\x80\x00\x00\x80\x80\x00\x00\xc0\xc0\xc0\xc0\xa6\xa6\x05\x05\n\n\x0e\x0e\x10\x10\x11\x11\x13\x13\x15\x15\x17\x17\x1a\x1a\x1b\x1b\x1d\x1d\x1e\x1e  !!$$&&((**++--//0033556688::;;==??AACCDDFFGGIIJJLLOOPPQQSSUUVVXXZZ\\\\^^__aabbcceegghhkkllmmoopprrssuuwwyy{{||}}\x7f\x7f\x81\x81\x82\x82\x83\x83\x84\x84\x86\x86\x88\x88\x89\x89\x8b\x8b\x8c\x8c\x8d\x8d\x8f\x8f\x90\x90\x91\x91\x92\x92\x94\x94\x96\x96\x97\x97\x98\x98\x9a\x9a\x9c\x9c\x9d\x9d\x9e\x9e\x9f\x9f\xa1\xa1\xa3\xa3\xa4\xa4\xa5\xa5\xa6\xa6\xa7\xa7\xa8\xa8\xa9\xa9\xaa\xaa\xab\xab\xab\xab\xac\xac\xad\xad\xae\xae\xaf\xaf\xaf\xaf\xb1\xb1\xb2\xb2\xb3\xb3\xb4\xb4\xb5\xb5\xb6\xb6\xb7\xb7\xb7\xb7\xb8\xb8\xba\xba\xbb\xbb\xbb\xbb\xbc\xbc\xbd\xbd\xbe\xbe\xc0\xc0\xc1\xc1\xc2\xc2\xc3\xc3\xc3\xc3\xc4\xc4\xc5\xc5\xc7\xc7\xc8\xc8\xc8\xc8\xc9\xc9\xca\xca\xcb\xcb\xcc\xcc\xce\xce\xcf\xcf\xd0\xd0\xd1\xd1\xd2\xd2\xd3\xd3\xd4\xd4\xd5\xd5\xd5\xd5\xd7\xd7\xd8\xd8\xd8\xd8\xd9\xd9\xda\xda\xdb\xdb\xdc\xdc\xde\xde\xdf\xdf\xe0\xe0\xe1\xe1\xe1\xe1\xe3\xe3\xe4\xe4\xe5\xe5\xe5\xe5\xe6\xe6\xe7\xe7\xe8\xe8\xe9\xe9\xea\xea\xeb\xeb\xeb\xeb\xec\xec\xec\xec\xed\xed\xed\xed\xed\xed\xee\xee\xee\xee\xef\xef\xef\xef\xef\xef\xf0\xf0\xf0\xf0\xf1\xf1\xf1\xf1\xf2\xf2\xf2\xf2\xf3\xf3\xf3\xf3\xf3\xf3\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf5\xf5\xf5\xf5\xf5\xf5\xf6\xf6\xf6\xf6\xf6\xf6\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf8\xf8\xf9\xf9\xf9\xf9\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfb\xfb\xfb\xfb\xfb\xfb\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\x00\x00\xff\xff\xa0\xa0\x80\x80\xff\xff\x00\x00\xff\xff\x00\x00\xff\xff\x00\x00\xff\xff'
    g_lut = b'\x00\x00\x00\x00\x80\x80\x80\x80\x00\x00\x00\x00\x80\x80\xc0\xc0\xdc\xdc\xca\xca\x01\x01\x01\x01\x01\x01\x01\x01\x02\x02\x02\x02\x02\x02\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x04\x04\x04\x04\x04\x04\x04\x04\x05\x05\x05\x05\x05\x05\x06\x06\x06\x06\x06\x06\x07\x07\x08\x08\x08\x08\t\t\n\n\n\n\x0b\x0b\r\r\x0e\x0e\x0e\x0e\x0f\x0f\x10\x10\x11\x11\x11\x11\x12\x12\x13\x13\x14\x14\x15\x15\x16\x16\x18\x18\x19\x19\x19\x19\x1a\x1a\x1b\x1b\x1c\x1c\x1d\x1d\x1d\x1d\x1e\x1e\x1f\x1f  ""$$%%&&\'\'(())**++,,--..001122445566778899::;;<<>>??AABBCCDDEEFFHHIIIIKKMMNNOOQQRRSSTTUUWWYY[[]]^^``bbccddeeggiikkllnnppqqssttvvwwzz{{{{}}~~\x7f\x7f\x81\x81\x82\x82\x83\x83\x85\x85\x86\x86\x88\x88\x89\x89\x8a\x8a\x8b\x8b\x8c\x8c\x8e\x8e\x8f\x8f\x91\x91\x92\x92\x93\x93\x95\x95\x97\x97\x98\x98\x99\x99\x9a\x9a\x9c\x9c\x9d\x9d\x9e\x9e\xa0\xa0\xa1\xa1\xa2\xa2\xa4\xa4\xa6\xa6\xa7\xa7\xa8\xa8\xaa\xaa\xab\xab\xac\xac\xad\xad\xaf\xaf\xb1\xb1\xb3\xb3\xb4\xb4\xb5\xb5\xb6\xb6\xb8\xb8\xb9\xb9\xba\xba\xbb\xbb\xbd\xbd\xbe\xbe\xc0\xc0\xc2\xc2\xc3\xc3\xc4\xc4\xc5\xc5\xc7\xc7\xc8\xc8\xc9\xc9\xca\xca\xcb\xcb\xcd\xcd\xce\xce\xd0\xd0\xd2\xd2\xd3\xd3\xd4\xd4\xd5\xd5\xd6\xd6\xd7\xd7\xd9\xd9\xd9\xd9\xda\xda\xda\xda\xdc\xdc\xde\xde\xde\xde\xdf\xdf\xe0\xe0\xe1\xe1\xe2\xe2\xe2\xe2\xe2\xe2\xe3\xe3\xe4\xe4\xe4\xe4\xe5\xe5\xe7\xe7\xe8\xe8\xe8\xe8\xe9\xe9\xeb\xeb\xec\xec\xec\xec\xed\xed\xed\xed\xee\xee\xef\xef\xef\xef\xf0\xf0\xf1\xf1\xf2\xf2\xf2\xf2\xf3\xf3\xf3\xf3\xf4\xf4\xf4\xf4\xf5\xf5\xf6\xf6\xf6\xf6\xf7\xf7\xf9\xf9\xfa\xfa\xfa\xfa\xfa\xfa\xfb\xfb\xfc\xfc\x00\x00\xfb\xfb\xa0\xa0\x80\x80\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\xff\xff\xff\xff'
    b_lut = b'\x00\x00\x00\x00\x00\x00\x00\x00\x80\x80\x80\x80\x80\x80\xc0\xc0\xc0\xc0\xf0\xf0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x02\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x05\x05\x05\x05\x05\x05\x05\x05\x05\x05\x06\x06\x06\x06\x06\x06\x06\x06\x06\x06\x07\x07\x08\x08\x08\x08\x08\x08\t\t\t\t\t\t\n\n\n\n\n\n\x0b\x0b\x0b\x0b\x0b\x0b\x0c\x0c\x0c\x0c\x0c\x0c\r\r\r\r\r\r\x0e\x0e\x0e\x0e\x0f\x0f\x0f\x0f\x0f\x0f\x10\x10\x10\x10\x10\x10\x11\x11\x11\x11\x12\x12\x12\x12\x13\x13\x14\x14\x14\x14\x15\x15\x16\x16\x16\x16\x17\x17\x18\x18\x18\x18\x18\x18\x19\x19\x19\x19\x1a\x1a\x1b\x1b\x1b\x1b\x1c\x1c\x1c\x1c\x1d\x1d\x1e\x1e\x1f\x1f    ""##$$&&\'\'(())))**++,,--//0011223344557799;;==??AADDFFHHKKLLOOQQSSVVWWYY[[^^``cceeggiimmpprruuyy||\x7f\x7f\x82\x82\x84\x84\x88\x88\x8b\x8b\x8d\x8d\x91\x91\x94\x94\x97\x97\x9a\x9a\x9d\x9d\xa0\xa0\xa2\xa2\xa6\xa6\xaa\xaa\xac\xac\xaf\xaf\xb3\xb3\xb4\xb4\xb7\xb7\xba\xba\xbc\xbc\xc0\xc0\xc2\xc2\xc5\xc5\xc8\xc8\xca\xca\xcd\xcd\xcf\xcf\xd1\xd1\xd4\xd4\xd7\xd7\xd9\xd9\xdc\xdc\xdf\xdf\xe2\xe2\xe4\xe4\xe7\xe7\xea\xea\xed\xed\x00\x00\xf0\xf0\xa4\xa4\x80\x80\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff'
    a_lut = None

    actual_depth = len(r_lut) / nr_entries * 8
    dtype = np.dtype("uint{:.0f}".format(actual_depth))

    for lut_bytes in [ii for ii in [r_lut, g_lut, b_lut, a_lut] if ii]:
        luts.append(np.frombuffer(lut_bytes, dtype=dtype))
    lut_lengths = [len(ii) for ii in luts]
    if not all(ii == lut_lengths[0] for ii in lut_lengths[1:]):
        raise ValueError("LUT data must be the same length")

    # IVs < `first_map` get set to first LUT entry (i.e. index 0)
    clipped_iv = np.zeros(arr.shape, dtype=dtype)
    # IVs >= `first_map` are mapped by the Palette Color LUTs
    # `first_map` may be negative, positive or 0
    mapped_pixels = arr >= first_map
    clipped_iv[mapped_pixels] = arr[mapped_pixels] - first_map
    # IVs > number of entries get set to last entry
    np.clip(clipped_iv, 0, nr_entries - 1, out=clipped_iv)

    # Output array may be RGB or RGBA
    out = np.empty(list(arr.shape) + [len(luts)], dtype=dtype)
    for ii, lut in enumerate(luts):
        out[..., ii] = lut[clipped_iv]

    return Image.fromarray(np.uint8(out / 257.))

def remove_old_data():
    """Remove old data from specified directories."""

    # Define the directories to clean
    dirs_to_clean = [
        config.FILE_DIR,
        config.ZIP_PATH,
        config.SAVE_DIR,
        config.TMP_PATH,
    ]

    for dir_path in dirs_to_clean:
        clean_directory(dir_path)

    #remove_empty_folders(config.TMP_PATH)


def clean_directory(dir_path):
    """Remove old files and directories from the specified path."""

    for data_path in listdir_fullpath(dir_path):
        if os.path.isfile(data_path):
            os.remove(data_path)
            continue
            '''elif os.path.isdir(data_path):
                try:
                    shutil.rmtree(data_path)
                except PermissionError:
                    continue'''


def listdir_fullpath(d):
    """Return a list of full paths for files and directories in the given directory."""
    return [os.path.join(d, f) for f in os.listdir(d)]


def check_if_data_old(data_path, time_tolerance=60):  # tolerance in minutes
    """Check if the data at the given path is older than the time_tolerance."""
    return time() - os.path.getmtime(data_path) > time_tolerance * 60


def remove_empty_folders(path_abs):
    """Remove empty folders with 'tmp' in their name from the given path."""
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0 and "tmp" in Path(path).stem:
            os.rmdir(path)


def multi_core_dicom_process(input): #refactor maybe if working
    frame, ds, save_path, i = input
    sw_frame = Image.fromarray(np.uint8(frame))  # Image.fromarray((x * 255).astype(np.uint8))
    sw_frame.save(os.path.join(save_path, str(i).zfill(4) + config.IMAGE_TYPE))

def multi_core_dicom(ds, save_path):
    pixel_array_numpy = ds.pixel_array
    ds_hash = hashlib.sha256(ds.pixel_array).hexdigest()
    with open(os.path.join(os.path.dirname(save_path), "hash.txt"), "w") as f:
        f.write(ds_hash)
    pool = ThreadPool(config.BATCH_SIZE)
    thread_array = [
        (frame, ds, save_path, i) for (i, frame) in enumerate(pixel_array_numpy)
    ]
    pool.map(multi_core_dicom_process, thread_array)

def process_zip(zip_file):
    zip_name = zip_file.name
    
    try:
        with zipfile.ZipFile(zip_name, 'r') as zip_file:
            save_path = os.path.join(config.FILE_DIR, Path(zip_name).stem,get_random_string(6))
            save_path_full = os.path.join(save_path, strftime("%Y-%m-%d-%H-%M", localtime()))
            save_path_raw = os.path.join(save_path_full, "raw_images")
            os.makedirs(save_path_raw)

            # Create a list of image file names in the ZIP file
            file_names = [f for f in zip_file.namelist() if not f.startswith("__MACOSX/") and not f.startswith(".") and "/." not in f and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff','.tif'))]
            
            # Define a worker function to process each image file
            def process_image(file_name):
                with zip_file.open(file_name) as file:
                    try:
                        image = Image.open(file)
                    except (IOError, SyntaxError, UnidentifiedImageError):
                        print("Skipping file:", file_name)
                        return
                    save_file_path = os.path.join(save_path_raw, Path(file_name).stem+config.IMAGE_TYPE)
                    image = convert_image(image)
                    image.save(save_file_path)
                    print("Saved image file:", save_file_path)

            # Process the image files in parallel using ThreadPool
            with ThreadPool(processes=config.MAX_THREADS) as pool:
                pool.map(process_image, file_names)
                    
    except zipfile.BadZipFile as e:
        print(e)
        raise FileNotFoundError(
            "Please wait for the upload to finish and try again."
        )
    return save_path_full, config.DEFAULT_SLICE_THICKNESS, config.DEFAULT_PIXEL_SPACING


def process_dicom(dicom_file):

    remove_old_data()
    try:
        ds = dicom.dcmread(dicom_file.name)
    except InvalidDicomError as e:
        print(e)
        raise FileNotFoundError(
            "Either upload was not finished or DICOM file oculd not be read. Try again once upload has finished."
        )

    patient_id = getattr(ds, "PatientID", None).replace("/", "_").replace(" ","").replace("#","")
    slice_thickness,pixel_spacing=get_slice_thickness_and_spacing(ds)

    timestamp = datetime.now().strftime("%H-%M-%S-%f")  # Replaces colons and dot
    save_path = os.path.join(config.FILE_DIR, f"{patient_id}_{timestamp}")

    if "None" in save_path:
        save_path = save_path.replace("None", get_random_string(6))
        
    save_path_raw = os.path.join(save_path, "raw_images")
    os.makedirs(save_path_raw)

    multi_core_dicom(ds, save_path_raw)
    return save_path,slice_thickness,pixel_spacing

def get_slice_thickness_and_spacing(ds):
    """
    Reads the slice thickness from an intravascular OCT DICOM file.
    :param filename: The name of the DICOM file to read.
    :return: The slice thickness in millimeters.
    """
    # Get the Pixel Measures Sequence from the DICOM file
    pixel_measures_sequence = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
    try:
        pixel_spacing= pixel_measures_sequence.PixelSpacing[0]
    except:
        pixel_spacing=config.DEFAULT_PIXEL_SPACING
    # Get the slice thickness from the Pixel Measures Sequence
    try:
        slice_thickness = pixel_measures_sequence.SliceThickness
    except:
        slice_thickness = config.DEFAULT_SLICE_THICKNESS
    return slice_thickness,pixel_spacing

def get_predictions_from_name(image_name, all_preds):
    predictions = []
    confidences = []

    for i in range(1, 5):
        search_name = str(image_name) + "_" + str(i)
        entry = all_preds.loc[search_name]
        prediction = entry["new_label"]
        center = [entry["centerx"], entry["centery"]]
        confidence = entry["new_confidence"]
        predictions.append(prediction)
        confidences.append(confidence)
        show_all_classes = entry["use_for_summary"]
    return predictions, confidences, center, bool(show_all_classes)

def get_unstented_indices(df,n=4,fill_gap=3):
    indices = []
    for i in range(0, len(df), n):
        row = df.iloc[i]
        name=int(row.name.split("_")[0])
        if row['nr_stents'] == 0:
            indices.append(name)
        elif i-n >=0 and i+n < len(df):
            if (row['nr_stents'] + df.iloc[i-n]['nr_stents'] + df.iloc[i+n]['nr_stents']) <= 3:
                indices.append(name)
        elif i-n < 0:
            if (row['nr_stents'] + df.iloc[i+n]['nr_stents']) <= 3:
                indices.append(name)
        elif i+n >= len(df):
            if (row['nr_stents'] + df.iloc[i-n]['nr_stents']) <= 3:
                indices.append(name)
    indices=fill_gaps(indices,fill_gap)
    return indices

def fill_gaps(arr, n):
    new_arr = []
    for i in range(0, len(arr)):
        diff_prev = int(arr[i]) - int(arr[i-1])
        if i==0:
            diff_prev=n+1
        if(i+1)<len(arr):
            diff_next = int(arr[i+1]) - int(arr[i])
        else:
            diff_next=n+1
        if diff_prev > n and diff_next > n:
            continue
        if diff_next<=n:
            for j in range(0, diff_next):
                new_arr.append(arr[i]+j)
        elif diff_prev<=n:
            new_arr.append(arr[i])
    return np.unique(new_arr)

def process_range_string(range_string, nr_images,predictions, neointima=True):
    if not "-" in range_string:
        use_indices = np.array(list(np.arange(0, nr_images)))
        if neointima:
            non_stented = get_unstented_indices(predictions)
            use_indices=np.setdiff1d(use_indices, non_stented)

    else:
        try:
            range_strings = range_string.replace(" ", "").split(",")
            use_indices = []
            for area in range_strings:
                split = area.split("-")
                start = np.max([0, int(split[0])])
                end = np.min([int(split[1]), nr_images - 1])
                use_indices += list(np.arange(start, end + 1))
        except Exception as e:
            print(e)
            use_indices = list(np.arange(0, nr_images))

    string_indeces = [str(s).zfill(4) for s in np.unique(use_indices)]
    range_string=get_intervals_as_string(string_indeces)

    return string_indeces,range_string

def get_intervals_as_string(arr):
    intervals = []
    arr=[int(i) for i in arr]
    start = arr[0]
    end = arr[0]
    for i in range(1, len(arr)):
        if arr[i] == end + 1:
            end = arr[i]
        else:
            intervals.append(str(start) + "-" + str(end))
            start = arr[i]
            end = arr[i]
    intervals.append(str(start) + "-" + str(end))
    return ', '.join(intervals)


def get_quadranted_images(
    images,
    masks,
    image_names,
    lumen_index=config.LUMEN_CLASS
):
    centers = []
    out_image_names = []
    quadranted = torch.Tensor([])

    for image, mask, image_name in zip(images, masks, image_names):
        quad_names = get_image_names(image_name)
        out_image_names.append(quad_names)
        center = get_center(mask, lumen_index)
        centers.append(center)
        im_crops = crop_at_center(image, center)
        quadranted = torch.cat((quadranted, im_crops), 0)

    return quadranted, centers, out_image_names


def get_center(mask, lumen_index):
    lumen = np.where(np.array(mask) == lumen_index)
    if (len(lumen[0])) > 0:
        center = [np.average(indices) for indices in lumen]
        center[0] = center[0] / len(mask[0])
        center[1] = center[1] / len(mask[1])
    else:
        center = [0.5, 0.5]
    return [center[1], center[0]]


def crop_at_center(image, center, target_size=config.IMAGE_SIZE_QUADRANT):

    center_image = [int(c) for c in np.array(center) * len(image[0])]
    image = torch.squeeze(torch.Tensor(image))

    quad_1_im = image[0 : center_image[1], center_image[0] :]
    quad_3_im = image[center_image[1] :, 0 : center_image[0]]
    quad_2_im = image[center_image[1] :, center_image[0] :]
    quad_4_im = image[0 : center_image[1], 0 : center_image[0]]

    quad_arr = [quad_1_im, quad_2_im, quad_3_im, quad_4_im]

    for i, im in enumerate(quad_arr):
        quad_arr[i] = torch.nn.functional.interpolate(
            torch.unsqueeze(torch.unsqueeze(im, 0), 0), size=target_size
        )

    return torch.cat(quad_arr, 0)

def get_class_volume_mm2(pred,pixel_spacing, class_nr):
    all_pixels = np.sum(np.where(pred == class_nr, 1, 0))
    if(pixel_spacing == 0):
        pixel_spacing = config.DEFAULT_PIXEL_SPACING

    return all_pixels*pixel_spacing *pixel_spacing 


def num_clusters(arr, n):
    clusters, _ = label(arr == n)
    return clusters.max()

def get_stent_nr(pred, stent_class=config.STENT_CLASS):
    labeled_array, _ = label(pred == stent_class)
    _, counts = np.unique(labeled_array, return_counts=True)
    thresholded = np.where(counts > 5, 1, 0)[
        1:
    ]  # first entry is background, we filter out stents smaller than 5 pixels
    return np.sum(thresholded)


def get_lumen_radius(mask,pixel_spacing, lumen_class=config.LUMEN_CLASS):
    all_pixels = np.sum(np.where(mask == lumen_class, 1, 0))
    
    if(pixel_spacing == 0):
        pixel_spacing = config.DEFAULT_PIXEL_SPACING
    
    pixel_to_mm=1/pixel_spacing*1024/mask.shape[0]
    radius = np.sqrt(all_pixels/np.pi)/pixel_to_mm #radius in pixels
    
    return radius


def get_neointima_thickness(mask, pixel_spacing, lumen_class=config.LUMEN_CLASS, neointima_class=config.NEOINTIMA_CLASS):
    all_pixels_r1 = np.sum(np.where(mask == lumen_class, 1, 0))
    all_pixels_r2 = np.sum(np.where(mask == neointima_class, 1, 0)) + all_pixels_r1
    
    if(pixel_spacing == 0):
        pixel_spacing = config.DEFAULT_PIXEL_SPACING
    
    pixel_to_mm=1/pixel_spacing*1024/mask.shape[0]

    r1 = np.sqrt(all_pixels_r1/np.pi)/pixel_to_mm
    r2 = np.sqrt(all_pixels_r2/np.pi)/pixel_to_mm

    return r2-r1


def get_image_names(im_name):
    # im_name=Path(im_name)
    return [im_name + "_" + str(i) for i in range(1, 5)]


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([("", arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([("", arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def get_smallest_lumen(image_path):
    prediction_file = pd.read_csv(os.path.join(image_path, "predictions.csv"))
    prediction_file = prediction_file[prediction_file.use_for_summary != 0]
    prediction_file = prediction_file.sort_values(
        "lumen radius", axis=0, ascending=True, na_position="last"
    )
    entry = prediction_file.iloc[0]

    return entry["image"].split("_")[0], entry["lumen radius"]


def get_bb_from_center(center, size, make_smaller=0):
    center = np.array(center) * size
    radius = np.min([center[0], size[0] - center[0], center[1], size[1] - center[1]])
    xy1 = center - radius + make_smaller
    xy2 = center + radius - make_smaller
    return [int(xy1[0]), int(xy1[1]), int(xy2[0]), int(xy2[1])], radius, center


def postprocess(predictions_df, factor=0.75):

    for name, entry in predictions_df.iterrows():
        index = name[-6:-2]
        try:
            after = predictions_df.loc[
                name.replace(index + "_", str(int(index) + 1).zfill(4) + "_")
            ]
        except:
            after = None
        try:
            before = predictions_df.loc[
                name.replace(index + "_", str(int(index) - 1).zfill(4) + "_")
            ]
        except:
            before = None

        tmp_distr = np.array(
            [
                entry["not determined"],
                entry["homogenous"],
                entry["non-homogenous"],
                entry["neoatherosclerosis"],
            ]
        )
        count = 0
        if after is not None and after["use_for_summary"]:
            count += 1
            tmp_distr += factor * np.array(
                [
                    after["not determined"],
                    after["homogenous"],
                    after["non-homogenous"],
                    after["neoatherosclerosis"],
                ]
            )
        if before is not None and before["use_for_summary"]:
            count += 1
            tmp_distr += factor * np.array(
                [
                    before["not determined"],
                    before["homogenous"],
                    before["non-homogenous"],
                    before["neoatherosclerosis"],
                ]
            )
        assert (
            len(tmp_distr) == 4
        ), "Something went wrong, distribution is too short or long " + str(index)
        predictions_df.at[name, "new_label"] = np.argmax(tmp_distr)
        predictions_df.at[name, "new_confidence"] = np.max(tmp_distr) / (
            1 + count * factor
        )
    return predictions_df


def summarize(folder_path, range_string, smallest_lumen_image):
    summary_dict = {}

    predictions_df = pd.read_csv(
        os.path.join(folder_path, "predictions.csv")
    ).set_index("image")

    filtered = predictions_df[predictions_df.use_for_summary == 1]

    predictions_count = {0: 0, 1: 0, 2: 0, 3: 0}
    predictions_relative = {0: 0, 1: 0, 2: 0, 3: 0}

    value_counts = filtered["new_label"].value_counts()
    total = 0.0

    for index in value_counts.index:
        predictions_count[index] = value_counts[index]
        total += value_counts[index]

    for index in value_counts.index:
        predictions_relative[index] = np.round(value_counts[index] / total, 3)

    # summary_dict["predictions_count"]=predictions_count
    summary_dict["Not determined neointima"] = predictions_relative[0]
    summary_dict["Homogenous neointima"] = predictions_relative[1]
    summary_dict["Nonhomogenous neointima"] = predictions_relative[2]
    summary_dict["Neoatherosclerotic neointima"] = predictions_relative[3]

    summary_dict["avg in-stent lumen radius"] = np.round(
        filtered["lumen radius"].mean(), 2
    )
    summary_dict["avg neointima thickness"] = np.round(filtered.neointima.mean(), 2)
    summary_dict["stented regions"] = range_string
    summary_dict["smallest lumen diameter"] = str(smallest_lumen_image)

    with open(os.path.join(folder_path, "summary.csv"), "w") as f:
        writer = csv.writer(f)

        for key, value in summary_dict.items():
            writer.writerow([key, str(value)])

    return create_summary_html(summary_dict_to_str(summary_dict))


def summary_dict_to_str(summary_dict):

    summary_dict["Not determined neointima"] = (
        str(np.round(summary_dict["Not determined neointima"] * 100, 1)) + " %"
    )
    summary_dict["Homogenous neointima"] = (
        str(np.round(summary_dict["Homogenous neointima"] * 100, 1)) + " %"
    )
    summary_dict["Nonhomogenous neointima"] = (
        str(np.round(summary_dict["Nonhomogenous neointima"] * 100, 1)) + " %"
    )
    summary_dict["Neoatherosclerotic neointima"] = (
        str(np.round(summary_dict["Neoatherosclerotic neointima"] * 100, 1)) + " %"
    )
    summary_dict["avg neointima thickness"]=(
        str(summary_dict["avg neointima thickness"]) + " mm"
    )
    summary_dict["avg in-stent lumen radius"] =(
        str(summary_dict["avg in-stent lumen radius"])+" mm"
    )
    return summary_dict


def save_files_to_s3(filename):
    s3 = boto3.client("s3")
    s3.upload_file(
        Filename=filename,
        Bucket="deepneo-storage",
        Key=Path(filename).name,
    )