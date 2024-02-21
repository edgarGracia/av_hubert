import sys
import os
import argparse
from pathlib import Path
import cv2
import tempfile
import numpy as np

np.float=np.float64
np.int=np.int32
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig


class AVHubert:

    def __init__(self, package_path: str, model_path: str):
        sys.path.insert(0, package_path)
        import hubert_asr
        import hubert_pretraining
        import hubert
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        self._temp_data_dir = Path(tempfile.mkdtemp())
        self.gen_cfg = GenerationConfig(beam=20)
        models, self.saved_cfg, self.task = \
            checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.saved_cfg.task.max_sample_size = None
        self.models = [model.eval().cuda() for model in models]
        self.saved_cfg.task.modalities = ["video"]
        self.saved_cfg.task.data = str(self._temp_data_dir)
        self.saved_cfg.task.label_dir = str(self._temp_data_dir)
        self.task = tasks.setup_task(self.saved_cfg.task)
    
    def _decode_fn(self, x, generator):
        dictionary = self.task.target_dictionary
        symbols_ignore = generator.symbols_to_strip_from_output
        symbols_ignore.add(dictionary.pad())
        return self.task.datasets["test"].label_processors[0].decode(x, symbols_ignore)

    def _clear_data(self):
        for f in self._temp_data_dir.iterdir():
            os.remove(f)
    
    def predict(self, video_path):
        self._clear_data()
        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
        self._temp_data_dir.joinpath("test.tsv").write_text("".join(tsv_cont))
        self._temp_data_dir.joinpath("test.wrd").write_text("DUMMY\n")
        self.task.load_dataset("test", task_cfg=self.saved_cfg.task)
        generator = self.task.build_generator(self.models, self.gen_cfg)
        itr = self.task.get_batch_iterator(dataset=self.task.dataset("test")).next_epoch_itr(shuffle=False)
        sample = next(itr)
        sample = utils.move_to_cuda(sample)
        hypos = self.task.inference_step(generator, self.models, sample)
        hypo = hypos[0][0]["tokens"].int().cpu()
        hypo = self._decode_fn(hypo, generator)
        return hypo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "video",
        help="Input mouth video"
    )
    ap.add_argument(
        "--out-text",
        type=Path,
        help="Optional output transcript file",
    )
    args = ap.parse_args()
    
    speech = AVHubert("./av_hubert/avhubert", "./data/finetune-model.pt")

    hypo = speech.predict(args.video)
    print(f"\n{hypo}\n")

    if args.out_text is not None:
        args.out_text.write_text(hypo)