import argparse
import numpy as np
from ultralytics import YOLO


class YoloMouthCrop:

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image: np.ndarray) -> :
        return self.model(image, verbose=False)

    def crop_image(self, yolo_results, image: np.ndarray, size: int = 96) -> np.ndarray:
        print(type(yolo_results))
        # if not results:
        #     return None
        # result = results[0]
        # xmin, ymin, xmax, ymax = list(map(int, result.boxes.xyxy.cpu().numpy()[0]))
        # keypoints = result.keypoints.xy.cpu().numpy()[0]
        
        # # crop = frame[int(keypoints[2,1]):ymax,xmin:xmax, :]
        # face_w = (xmax-xmin)*0.8
        # lip_center_y = (keypoints[3,1] + keypoints[4,1])//2
        # lip_center_x = (keypoints[3,0] + keypoints[4,0])//2
        # crop = frame[
        #     int(lip_center_y-face_w/2):int(lip_center_y+face_w/2),
        #     int(lip_center_x-face_w/2):int(lip_center_x+face_w/2),
        # ]
        # crop = cv2.resize(crop, (size, size))
        # return crop



    

# def crop_video_yolo(input_video_path, output_video_path, yolo_model_path):
#     cap = cv2.VideoCapture(input_video_path)
#     crop_frames = []
#     # for frame in tqdm(videogen):
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         crop = crop_frame(np.array(frame), model)
#         if crop is None:
#             continue
#         crop_frames.append(crop)
#         print(".", end="", flush=True)
#     cap.release()
#     write_video_ffmpeg(crop_frames, output_video_path, "/usr/bin/ffmpeg")  
        



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input",
        help="Input video to crop"
    )
    ap.add_argument(
        "output",
        help="Output crop video",
    )
    ap.add_argument(
        "--model",
        help="Path to the YOLO model",
        default="yolov8n-face.pt"
    )
    args = ap.parse_args()
    crop_video_yolo(args.input, args.output, args.model)
    # face_predictor_path = "/content/data/misc/shape_predictor_68_face_landmarks.dat"
    # mean_face_path = "/content/data/misc/20words_mean_face.npy"
    # preprocess_video(args.input, args.output, face_predictor_path, mean_face_path)
