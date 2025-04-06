#ngrok http 7860
import os
import time
from glob import glob

import gradio as gr
import torch
from torch.utils.data import DataLoader
from PIL import Image

import config
from data_handler import process_dicom,process_zip
from dataset import InferenceDataset
from html_handler import create_html, image_html, image_html_embed, expand_visualization_html, collapse_visualization_html
from inference import inference_neointima, update_wrap
import torch.nn as nn

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Union, List
from segmentation_models_pytorch.decoders.unetplusplus.model import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
import numpy as np
import yaml
from botocore.exceptions import ClientError
IMAGE_SIZE_QUADRANT=224
IMAGE_SIZE_SEGMENTATION=512

CLASS_MODEL_PATH="./models/class_model_deep_neo.pth"
SEG_MODEL_PATH="./models/model_stent_end.pth"
ZIP_PATH=config.ZIP_PATH
NUM_CLASSES=4

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes,pooling=None, dropout=0.1, activation=None):
        pool=None
        if pooling=="avg":
            pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == 'max':
            pool=nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout_1 = nn.Dropout(p=dropout, inplace=False) if dropout else nn.Identity()
        linear_1 = nn.Linear(in_channels, 1000, bias=True)
        relu_1=nn.ReLU()
        dropout_2 = nn.Dropout(p=dropout, inplace=False) if dropout else nn.Identity()
        linear_classification = nn.Linear(1000, classes, bias=True)
        super().__init__(pool,flatten,dropout_1, linear_1, relu_1,dropout_2,linear_classification) #activation)

class Inference_Dataset(Dataset):
    def __init__(self,images,num_classes = NUM_CLASSES):
        self.images = images
        self.num_classes=num_classes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        
        image_path=Path(self.images[idx])
        result = {}
        im = Image.open(image_path).resize((IMAGE_SIZE_SEGMENTATION,IMAGE_SIZE_SEGMENTATION), Image.NEAREST).convert("L")
        result["image"]=torch.unsqueeze(torch.Tensor(np.array(im)/255.),0)
        result["filename"] = str(image_path.stem)
        result["index"]=idx
        
        return result
    
class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        x = x.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        features = self.encoder(x)

        if(hasattr(self, "segmentation_head")):
           if( self.segmentation_head is not None):
                decoder_output = self.decoder(features)
                masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return labels

        return masks

    def predict(self, x):
        
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
    

class UnetPlusPlus(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = NUM_CLASSES,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], classes=4, **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()
        
def get_model(encoder_name="resnet18",pooling="max",classification=True,decoder_channels=(256,128,64,32,16)):
    
    attention=None
    
    aux_params=None
    if classification:
        aux_params=dict(pooling=pooling,
        dropout=0.2)

    model = UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        decoder_channels = decoder_channels,
        in_channels=1,
        classes=NUM_CLASSES,
        encoder_depth=5,
        decoder_attention_type=attention,
        aux_params=aux_params)

    if(classification):
        del model.segmentation_head
        del model.decoder

    return model

def load_model_dict(model_path,model,eval=True):
    
    if(not torch.cuda.is_available()):
        loaded=torch.load(model_path,map_location=torch.device('cpu'))
    else:
        loaded=torch.load(model_path)

    model.load_state_dict(loaded, strict = False)
    if eval:
        model.eval()

    return model


seg_model=get_model(classification=False)
class_model=get_model()

SEG_MODEL=load_model_dict(SEG_MODEL_PATH,seg_model)

CLASS_MODEL=load_model_dict(CLASS_MODEL_PATH,class_model)

def main(dicom_file, stent_range):
    # start_time=time.time()
    try:
        if '.zip' in dicom_file.name:
            image_path,slice_thickness_val,pixel_spacing_val=process_zip(dicom_file)
        else:
            image_path,slice_thickness_val,pixel_spacing_val = process_dicom(dicom_file)
        a = os.path.join(image_path, "raw_images")
        all_images = sorted(
            glob(
                os.path.join(image_path, "raw_images")
                + "/*"
                + config.IMAGE_TYPE
            )
        )

        dataset = InferenceDataset(all_images)
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        vis_out, image_path_gr, html_visualization, nr_images, html_summary = inference_neointima(
            stent_range, dataloader, SEG_MODEL, CLASS_MODEL, image_path,slice_thickness_val,pixel_spacing_val)
        Legend = gr.HTML(visible=False)
        legend_path = os.path.join(config.STYLE_PATH, "legend.png")
        
        return (
            vis_out,
            image_path_gr,
            html_visualization,
            nr_images,
            gr.update(visible=True),  #update
            gr.update(visible=False), #seg
            html_summary,
            gr.update(visible=True),
            1,
            pixel_spacing_val,
            slice_thickness_val,
            gr.update(value=image_html_embed(legend_path), visible=True),
            gr.update(elem_classes="expanded-html")
        )
    
    except ClientError as e:
        print(e)
        raise FileNotFoundError(
            "File not found. Please upload first and allow some time for the upload to finish."
        )

def change_vals(val):
    return (
        gr.update(maximum=val),
        gr.update(maximum=val),
        gr.update(maximum=val),
        gr.update(maximum=val),
    )

with gr.Blocks(title="DeepNeo", fill_width=True) as demo:
    with gr.Row():
        with gr.Column(min_width=500):
            gr.HTML(f"<style>{open(config.CSS_FILE).read()}</style>")
            gr.HTML("<script>window.onload = () => { document.title = 'DeepNeo'; }</script>")

            image_path_gr = gr.State("")
            nr_updates = gr.State(0)
            nr_images = gr.Number(value=1, interactive=True, visible=False)
            timestamp_gr = gr.State(value="")
            summary_html = gr.State(value="")
            pixel_spacing = gr.State(value=None)
            slice_thickness = gr.State(value=None)

            init_logo = gr.Image(
                value=os.path.join(config.STYLE_PATH, "logos/deepneo_logo_plus.png"),
                show_label=False,
                show_download_button=False,
            )

            with gr.Row() as initial_row:
                clinical_use_checkbox = gr.Checkbox(
                    label="This Software is strictly for academic research use only and it is not approved for clinical, diagnostic or treatment purposes. Please accept to proceed.",
                    elem_classes="checkbox"
                )

            with gr.Row() as accept_row:
                init_button = gr.Button("Accept")

            with gr.Row(visible=False) as secondary_row:
                inp = gr.File(label='Upload pullback')

                with gr.Tab("Legend"):
                    Legend = gr.HTML(visible=False)

                with gr.Tab("Download"):
                    vis_out = gr.File(label="Download")

                with gr.Tab("Info"):
                    info = gr.Text("Input either a DICOM or a .zip file ", label="")

                logo = gr.Image(
                    value=os.path.join(config.STYLE_PATH, "logos/deepneo_logo_plus.png"),
                    show_label=False,
                    show_download_button=False,
                    min_width=300,
                )

            with gr.Row(visible=False) as visualization_row:
                with gr.Column(min_width=1000):
                    html_visualization = gr.HTML(elem_id="viz_html", elem_classes="collapsed-html")

            # Sliders
            slider0 = gr.Slider(label="Frame", maximum=1, step=1, visible=False, elem_id="sliderelement0")
            slider1 = gr.Slider(label="Frame", maximum=1, step=1, visible=False, elem_id="sliderelement1")
            slider2 = gr.Slider(label="Frame", maximum=1, step=1, visible=False, elem_id="sliderelement2")
            slider3 = gr.Slider(label="Frame", maximum=1, step=1, visible=False, elem_id="sliderelement3")

            with gr.Row(visible=False) as another_row:
                with gr.Column(visible=False, min_width=500) as button_col:
                    with gr.Row():
                        btn_run_seg = gr.Button("Neointima analysis", visible=True)
                        button_update = gr.Button("Update", visible=False)

                with gr.Column(min_width=500):
                    stent_range = gr.Textbox(max_lines=1, label="Manually input range", visible=False)

            # Init function
            def initialize_neointima(checkbox):
                if checkbox:
                    timestamp = time.time_ns()
                    return {
                        timestamp_gr: timestamp,
                        secondary_row: gr.update(visible=True),
                        inp: gr.update(visible=True),
                        stent_range: gr.update(visible=True),
                        button_col: gr.update(visible=True),
                        visualization_row: gr.update(visible=True),
                        init_logo: gr.update(visible=False),
                        initial_row: gr.update(visible=False),
                        accept_row: gr.update(visible=False),
                        another_row: gr.update(visible=True),
                    }
                else:
                    raise ValueError("Please accept the terms to proceed.")

            init_button.click(
                fn=initialize_neointima,
                inputs=[clinical_use_checkbox],
                outputs=[
                    timestamp_gr,
                    secondary_row,
                    inp,
                    stent_range,
                    button_col,
                    visualization_row,
                    init_logo,
                    initial_row,
                    accept_row,
                    another_row,
                ],
            )

            nr_images.change(
                fn=change_vals,
                inputs=nr_images,
                outputs=[slider0, slider1, slider2, slider3],
            )

            btn_run_seg.click(
                fn=main,
                inputs=[inp, stent_range],
                outputs=[
                    vis_out,
                    image_path_gr,
                    html_visualization,
                    nr_images,
                    button_update,
                    btn_run_seg,
                    summary_html,
                    slider0,
                    nr_updates,
                    pixel_spacing,
                    slice_thickness,
                    Legend,
                    html_visualization,
                ],
            )

            button_update.click(
                fn=update_wrap,
                inputs=[stent_range, image_path_gr, nr_updates, slice_thickness, pixel_spacing],
                outputs=[
                    vis_out,
                    html_visualization,
                    summary_html,
                    nr_updates,
                    slider0,
                    slider1,
                    slider2,
                    slider3,
                ],
            )

            slider0.change(create_html, inputs=[image_path_gr, summary_html, slider0], outputs=html_visualization)
            slider1.change(create_html, inputs=[image_path_gr, summary_html, slider1], outputs=html_visualization)
            slider2.change(create_html, inputs=[image_path_gr, summary_html, slider2], outputs=html_visualization)
            slider3.change(create_html, inputs=[image_path_gr, summary_html, slider3], outputs=html_visualization)


demo.launch(
    server_name="0.0.0.0",
    show_error=True,
    favicon_path=os.path.join(config.STYLE_PATH, "deepneo_small.png"),
    show_api=False,
    share=True,
)
