#segmentation classes for neointima segmentation
LUMEN_CLASS=1
STENT_CLASS=2
NEOINTIMA_CLASS=3

#segmentation classes for calc and lesion segmentation
LESION_CLASS=1
CALC_CLASS=2
LUMEN_CLASS_CALC=3
DEFAULT_PIXEL_SPACING=6.842619e-003
DEFAULT_SLICE_THICKNESS=0.199999963
FILE_DIR="/home/ubuntu/DeepNeo/app/data"
APP_DIR="/home/ubuntu/DeepNeo/app/"
BATCH_SIZE=2
IMAGE_SIZE_QUADRANT=224
IMAGE_SIZE_SEGMENTATION=512
IMAGE_TYPE=".tif"
CLASS_MODEL_PATH="/home/ubuntu/DeepNeo/models/class_model_deep_neo.pth"
SEG_MODEL_PATH="/home/ubuntu/DeepNeo/models/model_stent_end.pth"
CALC_MODEL_PATH="/home/ubuntu/DeepNeo/models/model_state_dict_calc.pth"
ZIP_PATH="/home/ubuntu/DeepNeo/app/zipped_data/"
TMP_PATH="/tmp/"
SAVE_DIR="/home/ubuntu/DeepNeo/app/data"
CSS_FILE="/home/ubuntu/DeepNeo/app/styles/style.css"

ACCES_KEY="AKIAYARXXM5OQGFGLWC2"
SECRET_KEY="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"
    