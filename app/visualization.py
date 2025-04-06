import os
import time
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2

import config
from data_handler import get_bb_from_center, get_predictions_from_name,colour_np_array


def colour_mask(mask,show_all_classes=True):
    if show_all_classes:
        colours=np.array([[150., 150., 150.],#0 gray background
                             [100., 100., 100.], #1 dark gray lumen
                             [0., 0., 0.],# black stent
                             [200., 200., 200.],#3 light gray neointima
                             ])
    else:
        colours= np.array([[150., 150., 150.],#0 grau background
                             [100., 100., 100.], #1 
                             [150., 150., 150.],#2 
                             [150., 150., 150.],#3 
                             ])
    rgb_im=np.stack((mask,)*3, axis=-1)
    for i in range(len(colours)):
        rgb_im=np.where(rgb_im==(i,i,i),colours[i],rgb_im)
    return rgb_im


def draw_arcs(img, label,confidence, xy,i,alpha=255):
    
    color_dict = {0:(51, 153, 255,alpha), 1: (90, 234,19,alpha), 2: (255, 178, 25,alpha),3: (227, 8, 8,alpha)}
    xy=[(xy[0],xy[1]),(xy[2],xy[3])]
    img1 = ImageDraw.Draw(img)
    img1.arc(xy, start = 270+i*90, 
                end =  360+i*90, 
                fill = color_dict[label], 
                width = int(np.round(2*np.exp(confidence*2.5))))
    return img

def colour_image_border(image_path):
    color = "green"
    border = (10, 10, 10, 10)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Overwrite the border with a rectangle
    draw.rectangle((border[0]-1, border[1]-1, image.width-border[2], image.height-border[3]), outline=color)

    image.save(image_path)


def multi_core_vis_neointima(inputs):
    image,all_preds,mask,folder_path=inputs
    quad_predictions,quad_confs,center,show_all_classes=get_predictions_from_name(image.stem,all_preds)
    visualization_quadrants(mask, quad_predictions,quad_confs,center,image,folder_path,show_all_classes)

def visualization_quadrants(mask_path, quad_preds,quad_confs,center,image_path,save_path,show_all_classes,size=(512,512),offset=20):
    
    image=colour_np_array(np.array(Image.open(image_path).resize(size,Image.Resampling.NEAREST)))
    image_name=Path(image_path).stem
    mask=np.array(Image.open(mask_path))    
    bounding_bob,radius,center=get_bb_from_center(center,size,make_smaller=0)
    if show_all_classes:
        for j,(class_pred,quad_conf) in enumerate(zip(quad_preds,quad_confs)):
            image=draw_arcs(image,class_pred,float(quad_conf),bounding_bob,j)
        
    image.save(os.path.join(save_path, "quadrant_predictions", image_name+".jpg"))

    coloured_mask=colour_mask(mask,show_all_classes)
    image_save_path=os.path.join(save_path, "masks_coloured", image_name+".png")
    Image.fromarray(coloured_mask.astype(np.uint8)).save(image_save_path)

def draw_schematic_view_quadrants(data,offset=5,height=30):

    # Determine the width and height of the image
    width = int(len(data)*offset/4)
    # Create a blank image
    image = Image.new('RGB', (width, height*4), 'white')
    draw = ImageDraw.Draw(image)

    # Iterate through the data and draw rectangles
    for name,d in data.iterrows():
        i,quad=get_image_and_quad_nr(name)
        y=quad*height
        x=i*offset
        if d["use_for_summary"]==1:
            draw.rectangle((x, y, x + 3, y + height-2), fill=get_colour(d["new_label"]))
        else:
            draw.rectangle((x, y, x + 3, y + height-2), fill="#cccccc")
    return image


def get_image_and_quad_nr(image_name):
    split=image_name.split("_")
    return int(split[0]),int(split[1])-1

def join_images(image_a, image_b):
    width = max(image_a.width, image_b.width)
    height = image_a.height + image_b.height
    result_image = Image.new('RGB', (width, height), 'white')
    result_image.paste(image_a, (0, 0))
    result_image.paste(image_b, (0, image_a.height))
    return result_image

def draw_schematic_view_neointima(mask_folder,dataframe,offset=5,height=50,lumen_class=config.LUMEN_CLASS,distance=10,neointima_class=config.NEOINTIMA_CLASS,factor_neointima=5,factor_lumen=5):

    all_masks = sorted(os.scandir(mask_folder), key=lambda x: x.name)
    width=len(all_masks)*offset
    image = np.ones((height+distance, width, 3), dtype=np.uint8)*255
    
    for i, mask_path in enumerate(all_masks):
        name=Path(mask_path).stem
        df_entry=dataframe.loc[name+"_1"]
        mask = np.array(Image.open(mask_path.path).convert('L'))
        lumen_area = np.sum(mask==lumen_class)/(mask.shape[0]*mask.shape[1]) # we could also read it from the dataframe entry. however it has a differnt unit (diameter) in the dataframe
        cutoff=int(mask.shape[0]*df_entry["centery"])
        
        y_height_lumen = np.min([int(height*lumen_area*factor_lumen),height])
        
        y = int((height-y_height_lumen)/2)+distance
        x = i*offset
        
        image[y:y+y_height_lumen, x:x+offset] = (100,100,100)
        
        if df_entry["use_for_summary"]:
            upper_image=np.array([mask==neointima_class])[:,:cutoff]
            neointima_area_up=np.sum(upper_image)/(mask.shape[0]*mask.shape[1])
            neointima_area_down = np.sum(mask==neointima_class)/(mask.shape[0]*mask.shape[1])-neointima_area_up

            y_height_neointima_top = int(height*neointima_area_up*factor_neointima)
            y_height_neointima_bot = int(height*neointima_area_down*factor_neointima)

            y_neointima_top=max(distance,y-y_height_neointima_top)
            y_neointima_bottom=min(height+distance,y+y_height_lumen+y_height_neointima_bot)

            image[y_neointima_top:y, x:x+offset] = (200,200,200)
            image[y+y_height_lumen:y_neointima_bottom, x:x+offset] = (200,200,200)
        
    return Image.fromarray(image)


#------------------

def mark_and_draw(max_thickness):
    if max_thickness["value"]>0:
        image_path,point1,point2=max_thickness["im_path"].replace("masks","masks_coloured"),max_thickness["points"][0],max_thickness["points"][1]
        # Load the image and convert it to a NumPy array
        image_path=image_path.replace(config.IMAGE_TYPE,".png")
        image = cv2.imread(image_path)

        # Convert the points to tuples of integers
        pt1 = tuple(map(int, reversed(point1)))
        pt2 = tuple(map(int, reversed(point2)))

        # Draw circles at the points and a line between them
        cv2.circle(image, pt1, 2, (0, 0, 0), -1)
        cv2.circle(image, pt2, 2, (0, 0, 0), -1)
        cv2.line(image, pt1, pt2, (0, 0, 0), 1)
        cv2.imwrite(image_path, image)
    # Display the image with the marked points and line

def visualize_all(save_path):
    
    all_images=sorted(Path(os.path.join(save_path,"raw_images")).glob("*"+config.IMAGE_TYPE))
    all_masks=sorted(Path(os.path.join(save_path,"masks")).glob("*"+config.IMAGE_TYPE))
    time_0=time.time()
    pool = ThreadPool(4)
    all_preds=pd.read_csv(os.path.join(save_path,"predictions.csv")).set_index("image")
    thread_array=[(image,all_preds,mask,save_path) for (image,mask) in zip (all_images,all_masks)]
    pool.map(multi_core_vis_neointima, thread_array)

    print("time multiprocess visualization:",time.time()-time_0)

def get_colour(label):
    return ["#3399ffff","#5aea13ff","#ffb219ff","#e30808ff"][int(label)]

def draw_final_schematic_view(prediction_df,mask_folder):

    with ThreadPoolExecutor(max_workers=config.BATCH_SIZE) as executor:
        #submit the function to the thread pool
        view_future = executor.submit(draw_schematic_view_quadrants, prediction_df)
        lumen_future = executor.submit(draw_schematic_view_neointima, mask_folder,prediction_df)
        #lumen_future = executor.submit(draw_schematic_view_lumen, mask_folder)
        
        #wait for the results

        overview_view = view_future.result()
        overview_lumen = lumen_future.result()

        # Convert arrays to images if they are not already
        if not isinstance(overview_view, Image.Image):
            overview_view = Image.fromarray(overview_view)
        if not isinstance(overview_lumen, Image.Image):
            overview_lumen = Image.fromarray(overview_lumen)

        # Concatenate the two images vertically
        total_height = overview_view.height + overview_lumen.height
        combined_image = Image.new('RGB', (overview_view.width, total_height))
        combined_image.paste(overview_view, (0, 0))
        combined_image.paste(overview_lumen, (0, overview_view.height))

        overview = combined_image

        return overview       

def calculate_shift(mask_a_line):
    lumen_indices = np.where(mask_a_line == 3)[0]  

    if len(lumen_indices) == 0:
        return 0  # No lumen found, no shift needed
    # Calculate center of lumen
    center_of_lumen = int(np.mean(lumen_indices))
    
    # Determine the shift needed to center the lumen
    center_of_frame = mask_a_line.shape[0] // 2
    
    shift_amount = center_of_frame - center_of_lumen
    
    return shift_amount

def circular_shift(mask_a_line, shift_amount):
    # Perform a circular shift
    return np.roll(mask_a_line, shift_amount, axis=0)

def a_line(mask, offset):
    if mask is not None:
        final = np.ones((256,))*4
        mask = np.array(mask)

        # lumen>calc>lesion>background
        priorities = [3, 2, 1, 0]
        
        for i, row in enumerate(mask):
            for value in priorities:
                if value in row:
                    final[i] = value
                    break

    else:
        final = np.zeros((256,))  # if not in region of interest leave values at 0
        
    return np.transpose(np.repeat([final], offset, axis=0))

def polarize_mask(img):
    img=np.array(img)
    
    value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    return polar_image

def a_line_old(mask,offset): #duplicate with a_line function in calc_scoring
    
    if mask is not None:
        polar_mask=polarize_mask(mask)
        arr_lesion=[int(1 in n) for n in polar_mask]
        arr_calc=[int(2 in n)*2 for n in polar_mask]
        arr_joined= np.max([arr_lesion,arr_calc],axis=0) 
        final=np.where(arr_joined==0,3,arr_joined) #make sure lumen is class 3
    else:
        final=np.zeros((256,)) #if not in region of interest leave values at 0
    return np.transpose(np.repeat([final],offset,axis=0))

def draw_schematic_view_lumen(mask_folder,offset=5,height=50,lumen_class=config.LUMEN_CLASS_CALC,distance=10,factor_lumen=5):

    all_masks = sorted(os.scandir(mask_folder), key=lambda x: x.name)
    width=len(all_masks)*offset
    image = np.ones((height+distance, width, 3), dtype=np.uint8)*255
    
    for i, mask_path in enumerate(all_masks):
        
        mask = np.array(Image.open(mask_path.path).convert('L'))
        lumen_area = np.sum(mask==lumen_class)/(mask.shape[0]*mask.shape[1])
        y_height_lumen = np.min([int(height*lumen_area*factor_lumen),height])
        
        y = int((height-y_height_lumen)/2)+distance
        x = i*offset
        
        image[y:y+y_height_lumen, x:x+offset] = (100,100,100)
        
    return Image.fromarray(image)

def mark_and_draw(max_thickness):
    if max_thickness["value"]>0:
        image_path,point1,point2=max_thickness["im_path"].replace("masks","masks_coloured"),max_thickness["points"][0],max_thickness["points"][1]
        # Load the image and convert it to a NumPy array
        image_path=image_path.replace(config.IMAGE_TYPE,".png")
        image = cv2.imread(image_path)

        # Convert the points to tuples of integers
        pt1 = tuple(map(int, reversed(point1)))
        pt2 = tuple(map(int, reversed(point2)))

        # Draw circles at the points and a line between them
        cv2.circle(image, pt1, 2, (0, 0, 0), -1)
        cv2.circle(image, pt2, 2, (0, 0, 0), -1)
        cv2.line(image, pt1, pt2, (0, 0, 0), 1)
        cv2.imwrite(image_path, image)