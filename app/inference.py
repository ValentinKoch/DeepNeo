import csv
import os
import shutil
import time
from glob import glob

import config
import gradio as gr
import numpy as np
import pandas as pd
import torch
import ttach as tta
from PIL import Image
import torch.nn.functional as F

from data_handler import (get_lumen_radius, get_neointima_thickness,
                          get_quadranted_images, get_smallest_lumen,
                          get_stent_nr, postprocess, process_range_string,
                          save_files_to_s3, summarize,get_class_volume_mm2)
from html_handler import create_html
from visualization import (colour_image_border, mark_and_draw,
                           draw_final_schematic_view, visualize_all)


def update_slider(nr_updates):
    update_arr = [
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    ]
    update_arr[np.min([nr_updates, 3])] = gr.update(visible=True)
    return update_arr


def update_wrap(range_string, save_path, nr_updates, slice_thickness, pixel_spacing): 
    
    file_path, _, html, _, html_summary= update_neointima(range_string, save_path, nr_updates)
    s0, s1, s2, s3 = update_slider(nr_updates)
    return file_path, html, html_summary, nr_updates + 1, s0, s1, s2, s3

def update_neointima(range_string, save_path, nr_updates):
    folder_name = os.path.basename(os.path.normpath(save_path))
    prediction_df = pd.read_csv(os.path.join(save_path, "predictions.csv")).set_index(
        "image"
    )
    prediction_df.loc[:, "use_for_summary"] = 0
    nr_images = int(len(prediction_df) / 4)
    string_indeces,range_string = process_range_string(range_string, nr_images,prediction_df,True)

    for index in string_indeces:
        for i in range(1, 5):
            name = index + "_" + str(i)
            prediction_df.at[name, "use_for_summary"] = 1

    prediction_df = postprocess(prediction_df)
    prediction_df.to_csv(os.path.join(save_path, "predictions.csv"))
    visualize_all(save_path)

    smallest_lumen_image, lumen_area = get_smallest_lumen(save_path)
    im_path_seg = os.path.join(
        save_path, "masks_coloured", smallest_lumen_image + ".png"
    )
    im_path_class = os.path.join(
        save_path, "quadrant_predictions", smallest_lumen_image + ".jpg"
    )

    #colour_image_border(im_path_seg)

    zip_file_path = os.path.join(config.ZIP_PATH, folder_name)


    html_summary = summarize(save_path, range_string, smallest_lumen_image)


    shutil.make_archive(zip_file_path, "zip", save_path)

    schematic_view = draw_final_schematic_view(prediction_df,os.path.join(save_path,"masks"))
    schematic_view.save(os.path.join(save_path, "schematic.png"))
    schematic_view.save(
        os.path.join(config.FILE_DIR, "schema" + str(nr_updates) + ".png")
    )

    return (
        zip_file_path + ".zip",
        save_path,
        create_html(
            save_path,
            html_summary,
            True,
            im_path_seg,
            im_path_class,
            schematic_path=os.path.join(save_path, "schematic.png")
        ),
        nr_images - 1,
        html_summary,
    )


def inference_neointima(range_string, dataloader, seg_model, class_model, save_path, slice_thickness,pixel_spacing):
    mask_path = os.path.join(save_path, "masks")
    im_path_seg = os.path.join(save_path, "masks_coloured")
    os.makedirs(mask_path)
    os.makedirs(im_path_seg)
    os.makedirs(os.path.join(save_path, "quadrant_predictions"))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    seg_model = seg_model.to(device)
    class_model = class_model.to(device)

    print("Starting prediction..")
    time_0 = time.time()
    prediction_df_path = os.path.join(save_path, "predictions.csv")

    with open(prediction_df_path, "w") as f:
        writer = csv.writer(f)
        row = [
            "image",
            "not determined",
            "homogenous",
            "non-homogenous",
            "neoatherosclerosis",
            "prediction",
            "nr_stents",
            "lumen radius",
            "neointima",
            "centerx",
            "centery",
            "use_for_summary",
        ]
        writer.writerow(row)

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            masks = (
                np.argmax(seg_model(batch["image"]).cpu(), axis=1).numpy().astype(np.uint8)
            )
            quadranted_images, centers, quadrant_names_chunked = get_quadranted_images(
                batch["image"], masks, batch["filename"]
            )
            all_quadrant_class_predictions = get_test_time_augmented_predictions(
                class_model, quadranted_images
            )
        class_predictions = np.argmax(all_quadrant_class_predictions, axis=1)

        class_prediction_chunks = [
            class_predictions[x : x + 4] for x in range(0, len(class_predictions), 4)
        ]
        class_distributions = np.array(
            [
                q / p
                for q, p in zip(
                    all_quadrant_class_predictions,
                    np.sum(all_quadrant_class_predictions, axis=1),
                )
            ]
        )
        class_distribution_chunks = [
            class_distributions[x : x + 4]
            for x in range(0, len(class_distributions), 4)
        ]

        with open(prediction_df_path, "a") as f:
            writer = csv.writer(f)
            for (
                image,
                image_name,
                mask,
                center,
                quadrant_predictions,
                quadrant_distributions,
                quadrant_names,
            ) in zip(
                batch["image"],
                batch["filename"],
                masks,
                centers,
                class_prediction_chunks,
                class_distribution_chunks,
                quadrant_names_chunked,
            ):
                nr_stents = get_stent_nr(mask)
                # get_uncovered_stents(mask)
                lumen_radius = get_lumen_radius(mask,pixel_spacing)
                neointima = (
                    get_neointima_thickness(mask,pixel_spacing ,lumen_class=config.LUMEN_CLASS, neointima_class=config.NEOINTIMA_CLASS)

                )
                Image.fromarray(mask).save(
                    os.path.join(mask_path, image_name + config.IMAGE_TYPE)
                )
                for (
                    quadrant_name,
                    quadrant_class_prediction,
                    quadrant_class_distribution,
                ) in zip(quadrant_names, quadrant_predictions, quadrant_distributions):
                    row = [
                        quadrant_name,
                        *quadrant_class_distribution,
                        quadrant_class_prediction,
                        nr_stents,
                        lumen_radius,
                        neointima,
                        *center,
                        0,
                    ]
                    writer.writerow(row)

    print(time.time() - time_0, "s for prediction")
    print("Clearing cache")
    torch.cuda.empty_cache()
    return update_neointima(range_string, save_path, 0)


def move_files(source_path, target_path, file_type=config.IMAGE_TYPE):

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    all_files = glob(os.path.join(source_path, "*" + file_type))

    for file in all_files:
        shutil.move(file, target_path)

def get_test_time_augmented_predictions(model, image):
    softmax = torch.nn.Softmax(dim=1)
    labels = []
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]  # tta.Multiply(factors=[0.95, 1, 1.05])
    )
    for transform in transforms:
        augmented_image = transform.augment_image(image)
        model_output_label = model(augmented_image)
        softmax_output = F.softmax(model_output_label, dim=1)
        deaug_label = transform.deaugment_label(softmax_output)

        labels.append(deaug_label.cpu().detach().numpy()) 

    label = np.mean(np.array(labels) ** 0.5, axis=0)

    return label

def get_test_time_augmented_mask_predictions(model, image):
    softmax2d = torch.nn.Softmax2d()
    masks=[]
    transforms = tta.Compose(
    [
        tta.HorizontalFlip()
        #tta.Rotate90(angles=[0, 90, 180, 270])      
    ]
)
    for transformer in transforms: # custom transforms or e.g. tta.aliases.d4_transform() 
        
        # augment image
        augmented_image = transformer.augment_image(image)
        
        # pass to model
        model_output = softmax2d(model(augmented_image))
        
        # reverse augmentation for mask and label
        deaug_mask = transformer.deaugment_mask(model_output)

        masks.append(deaug_mask.cpu().numpy())
        
    # reduce results as you want, e.g mean/max/min
    mask = np.mean(masks,axis=0)
    return mask
