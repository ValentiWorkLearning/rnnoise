import os
import scipy
import librosa

import numpy as np
import onnxruntime as ort
from enum import Enum

import pathlib
import requests
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["SigMOS", "Version"]


class Version(Enum):
    V1 = "v1"  # 15.10.2023


class SigMOS:
    """
    MOS Estimator for the P.804 standard.
    See https://arxiv.org/pdf/2309.07385.pdf
    """

    def __init__(self, model_dir, model_version=Version.V1):
        assert model_version in [v for v in Version]

        model_path_history = {
            Version.V1: os.path.join(
                model_dir, "model-sigmos_1697718653_41d092e8-epo-200.onnx"
            )
        }

        self.sampling_rate = 48_000
        self.resample_type = "fft"
        self.model_version = model_version

        # STFT params
        self.dft_size = 960
        self.frame_size = 480
        self.window_length = 960
        self.window = np.sqrt(np.hanning(int(self.window_length) + 1)[:-1]).astype(
            np.float32
        )

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path_history[model_version], options)

    def stft(self, signal):
        last_frame = len(signal) % self.frame_size
        if last_frame == 0:
            last_frame = self.frame_size

        padded_signal = np.pad(
            signal,
            ((self.window_length - self.frame_size, self.window_length - last_frame),),
        )
        frames = librosa.util.frame(
            padded_signal,
            frame_length=len(self.window),
            hop_length=self.frame_size,
            axis=0,
        )
        spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
        return spec.astype(np.complex64)

    @staticmethod
    def compressed_mag_complex(x: np.ndarray, compress_factor=0.3):
        x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)
        x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
        if compress_factor == 1:
            mag = np.sqrt(x2)
        else:
            x = np.power(x2, (compress_factor - 1) / 2) * x
            mag = np.power(x2, compress_factor / 2)

        features = np.concatenate((mag, x), axis=-2)
        features = np.transpose(features, (1, 0, 2))
        return np.expand_dims(features, 0)

    def run(self, audio: np.ndarray, sr=None):
        if sr is not None and sr != self.sampling_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sampling_rate,
                res_type=self.resample_type,
            )
            print(f"Audio file resampled from {sr} to {self.sampling_rate}!")

        features = self.stft(audio)
        features = self.compressed_mag_complex(features)

        onnx_inputs = {inp.name: features for inp in self.session.get_inputs()}
        output = self.session.run(None, onnx_inputs)[0][0]

        result = {
            "MOS_COL": float(output[0]),
            "MOS_DISC": float(output[1]),
            "MOS_LOUD": float(output[2]),
            "MOS_NOISE": float(output[3]),
            "MOS_REVERB": float(output[4]),
            "MOS_SIG": float(output[5]),
            "MOS_OVRL": float(output[6]),
        }
        return result


class SigmosModelRunner:
    def __init__(self) -> None:
        self._model_directory = pathlib.Path(os.getcwd(), "sigmos_eval_model")
        self._prefetch_model_location = pathlib.Path(
            self._model_directory, "model-sigmos_1697718653_41d092e8-epo-200.onnx"
        )
        self._sigmos_evaluator = None

    def evaluate_file(self, audio_file_path: pathlib.Path):
        if not self._sigmos_evaluator:
            self._prefetch_model()
            self._sigmos_evaluator = SigMOS(self._model_directory)

        audio_data, _sample_rate = librosa.load(audio_file_path, sr=None)
        return self._sigmos_evaluator.run(audio_data)

    def _prefetch_model(self):
        RAW_MODEL_LINK = "https://github.com/microsoft/SIG-Challenge/raw/main/ICASSP2024/sigmos/model-sigmos_1697718653_41d092e8-epo-200.onnx"

        local_file_path = self._prefetch_model_location
        if local_file_path.exists():
            return

        response = requests.get(RAW_MODEL_LINK)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download file. Status code: {response.status_code}"
            )

        if not self._model_directory.exists():
            os.makedirs(self._model_directory)

        with open(local_file_path, "wb") as file:
            file.write(response.content)
            print(f"File downloaded successfully and saved to {local_file_path}")


@dataclass
class SigmosResultItem:
    sigmos_data: dict
    audio_path: pathlib.Path


def plot_sigmos_results(
    sigmos_items: list[SigmosResultItem],
    evaluation_title: str,
    output_path: pathlib.Path,
):
    competitors = [sigmos_item.sigmos_data for sigmos_item in sigmos_items]
    sigmos_dataframe = pd.DataFrame(competitors)
    transposed_frame = sigmos_dataframe.T

    bar_chart = transposed_frame.plot.bar(width=0.8, figsize=(16, 5))
    legends = [sigmos_item.audio_path.stem for sigmos_item in sigmos_items]

    bar_chart.legend(legends)
    bar_chart.tick_params(axis="x", labelrotation=0)

    for container in bar_chart.containers:
        bar_chart.bar_label(container, rotation=90, label_type="center")

    plt.title(f"SIGMOS: {evaluation_title}")
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    """ 
        Sample code to run the SigMOS estimator. 
        V1 (current model) is an alpha version and should be used in accordance.
    """
    model_dir = r"."
    sigmos_estimator = SigMOS(model_dir=model_dir)

    # input data must have sr=48kHz, otherwise please specify the sr (it will be resampled to 48kHz internally)
    sampling_rate = 48_000
    dummy_data = np.random.rand(5 * sampling_rate)
    dummy_result = sigmos_estimator.run(dummy_data, sr=sampling_rate)
    print(dummy_result)
