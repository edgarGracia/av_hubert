import argparse
import os
import subprocess
import shutil
import tempfile
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO


def write_video_ffmpeg(rois, target_path, ffmpeg):
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir+'/%0'+str(decimals)+'d.png' + "'\n")
    # ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', target_path]
    pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    return


class YoloMouthCrop:

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image: np.ndarray) -> list:
        return self.model(image, verbose=False)

    def crop_image(self, yolo_results: list, image: np.ndarray, size: int = 96) -> np.ndarray:
        if not yolo_results:
            return None
        yolo_results = yolo_results[0]
        xmin, ymin, xmax, ymax = list(map(int, yolo_results.boxes.xyxy.cpu().numpy()[0]))
        keypoints = yolo_results.keypoints.xy.cpu().numpy()[0]
        
        face_w = (xmax-xmin)*0.8
        lip_center_y = (keypoints[3,1] + keypoints[4,1])//2
        lip_center_x = (keypoints[3,0] + keypoints[4,0])//2
        crop = image[
            int(lip_center_y-face_w/2):int(lip_center_y+face_w/2),
            int(lip_center_x-face_w/2):int(lip_center_x+face_w/2),
        ]
        crop = cv2.resize(crop, (size, size))
        return crop

    def crop_video(self, input_video_path, output_video_path, size: int = 96):
        cap = cv2.VideoCapture(input_video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        crop_frames = []
        pbar = tqdm(total=num_frames)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yolo_results = self.predict(frame)
            crop = self.crop_image(yolo_results, frame, size)
            if crop is not None:
                crop_frames.append(crop)
            pbar.update()
        cap.release()
        write_video_ffmpeg(crop_frames, output_video_path, "/usr/bin/ffmpeg")  


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input",
        help="Input video to crop"
    )
    ap.add_argument(
        "output",
        help="Output cropped video",
    )
    ap.add_argument(
        "--model",
        help="Path to the YOLO model",
        default="yolov8n-face.pt"
    )
    ap.add_argument(
        "--size",
        help="Crop size",
        default=96,
        type=int
    )
    args = ap.parse_args()
    yolo = YoloMouthCrop(args.model)
    yolo.crop_video(args.input, args.output, args.size)