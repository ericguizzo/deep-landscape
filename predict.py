import argparse
import logging
import os
import tempfile
from pathlib import Path

import cog
import yaml
import random

import constants
from logger import setup_logger
from model import NoiseChangeMode, StyleChangeMode
from utils import get_model, save_sample


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model"""
        config_path = "configs/train/256.yaml"
        iteration = None
        # iteration = 475000
        debug = False

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model_name = os.path.basename(config_path)[: -len(".yaml")]

        os.makedirs(constants.LOG_DIR, exist_ok=True)
        setup_logger(
            out_file=os.path.join(constants.LOG_DIR, "gen_" + model_name + ".log"),
            stdout_level=logging.DEBUG if debug else logging.INFO,
        )

        self.gen_model = get_model(
            model_name=model_name, config=config, iteration=iteration
        )
        self.config = config

    # @cog.input("input", type=Path, help="Input image path")
    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    def predict(self, seed):
        """Compute prediction"""

        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)

        output_path_dir = Path(tempfile.mkdtemp())
        output_path = output_path_dir / "homography/selected_homographies/475000.mp4"
        gen_path = str(output_path_dir)
        os.makedirs(gen_path, exist_ok=True)

        homography_dir = "homographies/selected_homographies"
        available_modes = set(
            ("homography", "noise_homography", "shift", "noise", "style", "images")
        )
        mode = "homography"  # UI
        num_files = 1
        image_n_frames = constants.IMAGE_N_FRAMES
        video_n_frames = constants.VIDEO_N_FRAMES
        inversed = False
        separate_files = False
        save_frames = False
        trunc = constants.TRUNCATION_PSI

        generator = self.gen_model["g_running"].eval()
        code_size = self.config.get("code_size", constants.DEFAULT_CODE_SIZE)
        alpha = self.gen_model["alpha"]
        step = self.gen_model["step"]
        resolution = self.gen_model["resolution"]
        iteration = self.gen_model["iteration"]

        assert mode in available_modes, mode
        if mode.startswith("noise"):
            style_change_mode = StyleChangeMode.REPEAT
        else:
            style_change_mode = StyleChangeMode.INTERPOLATE

        if mode == "style":
            noise_change_mode = NoiseChangeMode.FIXED
        elif mode.endswith("homography"):
            noise_change_mode = NoiseChangeMode.HOMOGRAPHY
            assert (
                homography_dir is not None
            ), "The homography mode needs a path to a homography directory!"
        else:
            noise_change_mode = NoiseChangeMode.SHIFT
        noise_change_modes = [noise_change_mode] * constants.MAX_LAYERS_NUM

        if mode == "images":
            save_video = False
            save_images = True
        else:
            save_video = True
            save_images = False

        save_dir = os.path.join(gen_path, mode)
        if mode.endswith("homography"):
            save_dir = os.path.join(save_dir, os.path.basename(homography_dir))

        save_sample(
            generator,
            alpha,
            step,
            code_size,
            resolution,
            save_dir=save_dir,
            name=("inversed_" if inversed else "") + str(iteration + 1).zfill(6),
            sample_size=constants.SAMPLE_SIZE,
            truncation_psi=trunc,
            images_n_frames=image_n_frames,
            video_n_frames=video_n_frames,
            save_images=save_images,
            save_video=save_video,
            style_change_mode=style_change_mode,
            noise_change_modes=noise_change_modes,
            inversed=inversed,
            homography_dir=homography_dir,
            separate_files=separate_files,
            num_files=num_files,
            save_frames=save_frames,
        )

        return output_path
