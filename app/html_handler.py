import os
from pathlib import Path
from PIL import Image
import base64


import config

def expand_visualization_html():
    return gr.update(elem_classes="expanded-html")

def collapse_visualization_html():
    return gr.update(elem_classes="collapsed-html")

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

def encode_base64(path):
        ext = Path(path).suffix.lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode()}"

def create_html(image_save_path, summary_html, i=0, im_path_seg=None, im_path_class=None, schematic_path=None):
    os.chdir(config.APP_DIR)
    if image_save_path == "":
        return ""

    schematic_path = os.path.join(image_save_path, "schematic.png")
    schematic_html = ""
    if schematic_path and os.path.exists(schematic_path):
        encoded_schematic = encode_base64(schematic_path)
        schematic_html = f'<div class="column"><img src="{encoded_schematic}" style=" height:auto; margin-top:5px;"></div>'
    
    if im_path_seg is None:
        all_ims_seg = Path(os.path.join(image_save_path, "masks_coloured")).glob("*.png")
        im_path_seg = sorted(all_ims_seg)[i]
    
    if im_path_class is None and os.path.exists(os.path.join(image_save_path, "images_coloured")):
        all_ims_class = list(Path(os.path.join(image_save_path, "images_coloured")).glob("*.jpg"))
        if len(all_ims_class) > 0:
            im_path_class = sorted(all_ims_class)[i]
    else:
        all_ims_class = list(Path(os.path.join(image_save_path, "quadrant_predictions")).glob("*.jpg"))
        if len(all_ims_class) > 0:
            im_path_class = sorted(all_ims_class)[i]

    encoded_class = encode_base64(im_path_class)
    encoded_seg = encode_base64(im_path_seg)

    html = f"""
    <div class="row">
        <div class="column">
            <img src="{encoded_class}" style="width:100%; max-width:500px; height:auto;">
        </div>
        <div class="column">
            <img src="{encoded_seg}" style="width:100%; max-width:500px; height:auto;">
        </div>
        <div class="column">
            {summary_html}
        </div>
    </div>
    {schematic_html}
    <p>{i}</p>
    """
    return html

def image_html(*image_paths):
    os.chdir(config.APP_DIR)
    html_string = "<html><body stye='color:white'>"
    for im_path in image_paths:
        relative_path = os.path.relpath(im_path, config.APP_DIR)
        processed_path = "./" + relative_path
        html_string += "<img src='" + processed_path + "'>"
    html_string += "</body></html>"
    return html_string

def image_html_embed(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" class="legend-img" />'


def schematic_html(im_path,alt="schematic"):
    os.chdir(config.APP_DIR)
    relative_path_class = os.path.relpath(im_path, config.APP_DIR)
    processed_im_path =  relative_path_class

    html = """ <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate, post-check=0, pre-check=0">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="-1">
    <img src='""" + processed_im_path + """?nocache=<?php echo time(); ?>' width='100%' alt='"""+alt+"""'>"""
    return html


