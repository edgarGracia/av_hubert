mkdir -p data/

# AV-HuBERT weights from https://facebookresearch.github.io/av_hubert/
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt -O data/finetune-model.pt

# YOLOv8 Face detector + keypoints from https://github.com/derronqi/yolov8-face?tab=readme-ov-file 
wget "https://drive.usercontent.google.com/u/0/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb&export=download" -O data/yolov8n-face.pt