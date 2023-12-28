import os
from pathlib import Path
from PIL import Image

import config

def create_summary_html(summary_dict):
    tmp_str = ""
    for key, value in summary_dict.items():
        tmp_str += '<tr><td style="color:#000000" >&nbsp;&nbsp;&nbsp;' + str(key) + "</td><td style='color:#000000'>" + str(value) + "</td></tr>"
    return (
        """<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <div style="color:#000000"><table class="styled-table">
        <thead>
            <tr>
                <th style="color:#000000">&nbsp;&nbsp;&nbsp;Stat</th>
                <th style="color:#000000">Value</th>
            </tr>
        </thead>
        <tbody>
            """
        + tmp_str
        + """
        </tbody>
    </table></div>"""
    )


def create_html(
    image_save_path, summary_html, i=0, im_path_seg=None, im_path_class=None
):
    os.chdir(config.APP_DIR)  # not sure at what point the working dir changes..
    if image_save_path == "":
        return ""
    if im_path_seg is None:
        all_ims_seg = Path(os.path.join(image_save_path, "masks_coloured")).glob(
            "*.png"
        )

        im_path_seg = sorted(all_ims_seg)[i]
    
    if im_path_class is None and os.path.exists(os.path.join(image_save_path, "images_coloured")):
        all_ims_class = list(Path(os.path.join(image_save_path, "images_coloured")).glob("*.jpg"))
        if len(list(all_ims_class)) > 0:
            im_path_class = sorted(list(all_ims_class))[i]
    
    else:
        all_ims_class = list(Path(os.path.join(image_save_path, "quadrant_predictions")).glob("*.jpg"))
        if len(list(all_ims_class)) > 0:
            im_path_class = sorted(list(all_ims_class))[i]

    relative_path_seg = os.path.relpath(im_path_seg, config.APP_DIR)
    processed_im_path_seg = "file/./" + relative_path_seg
    relative_path_class = os.path.relpath(im_path_class, config.APP_DIR)
    processed_im_path_class = "file/./" + relative_path_class


    html = (
        """
  <div class="row">
    <div class="column">
      <img src='"""
        + processed_im_path_class
        + """'style="width:100%">
    </div>
      <div class="column">
      <img src='"""
        + processed_im_path_seg
        + """'style="width:100%">
    </div>
    <div class="column">"""
        + summary_html
        + """</div>
  </div>"""
  """
  <p>"""
        + str(i)
        + """</p>
  """
    )


    return html

def create_html_calc(image_save_path, summary_html, i=0, im_path_seg=None, im_path_class=None, im_path_raw=None):
    os.chdir(config.APP_DIR)
    
    if image_save_path == "":
        return ""
    
    if im_path_seg is None:
        all_ims_seg = Path(os.path.join(image_save_path, "masks_coloured")).glob("*.png")
        im_path_seg = sorted(all_ims_seg)[i]
        
    if im_path_raw is None and os.path.exists(os.path.join(image_save_path, "images_coloured")):
        all_ims_raw = list(Path(os.path.join(image_save_path, "images_coloured")).glob("*.jpg"))
        if len(list(all_ims_raw)) > 0:
            im_path_raw = sorted(list(all_ims_raw))[i]
        else:
            all_ims_raw = list(Path(os.path.join(image_save_path, "images_coloured")).glob("*.jpg"))
            if len(list(all_ims_raw)) > 0:
                im_path_raw = sorted(list(all_ims_raw))[i]
    
    if im_path_raw is not None:
        relative_path_raw = os.path.relpath(im_path_raw, config.APP_DIR)
        processed_im_path_raw = "file/./" + relative_path_raw
    else:
        processed_im_path_raw = ""
        
    relative_path_seg = os.path.relpath(im_path_seg, config.APP_DIR)
    processed_im_path_seg = "file/./" + relative_path_seg

    html = """
        <div class="row">
            """ + ("""
            <div class="column">
                <img src='""" + processed_im_path_raw + """' style="width:100%">
            </div>
            """ if processed_im_path_raw else "") + """
            <div class="column">
                <img src='""" + processed_im_path_seg + """' style="width:100%">
                
            </div>
            <div class = "column> """ + summary_html + """ </div>
        </div>
        <p>""" + str(i) + """</p>
    """


    
    return html

def image_html(*image_paths):
    os.chdir(config.APP_DIR)
    html_string = "<html><body stye='color:white'>"
    for im_path in image_paths:
        relative_path = os.path.relpath(im_path, config.APP_DIR)
        processed_path = "file/./" + relative_path
        html_string += "<img src='" + processed_path + "'>"
    html_string += "</body></html>"
    return html_string


def schematic_html(im_path,alt="schematic"):
    os.chdir(config.APP_DIR)
    relative_path_class = os.path.relpath(im_path, config.APP_DIR)
    processed_im_path = "file/./" + relative_path_class

    html = """ <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate, post-check=0, pre-check=0">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="-1">
    <img src='""" + processed_im_path + """?nocache=<?php echo time(); ?>' width='100%' alt='"""+alt+"""'>"""
    return html


