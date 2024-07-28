import os
import sys
import argparse
import subprocess
import pathlib
import tempfile

import soundfile as sf
import librosa
import librosa.display
import shutil

import numpy as np
import matplotlib.pyplot as plt

from pyroomacoustics import denoise
import pyloudnorm as pyln

from sigmos_eval import SigmosModelRunner, SigmosResultItem, plot_sigmos_results

sys.path.append(os.path.join(os.getcwd(), "pyagc/agc"))


from pyagc.agc import tf_agc


def plot_spectrogram(
    file_path: pathlib.Path, corresponding_axes, figure, axis_title: str
):
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.stft(y, hop_length=256, n_fft=4096)

    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    spectrogram = librosa.display.specshow(
        S_db, sr=sr, hop_length=256, x_axis="time", y_axis="log", ax=corresponding_axes
    )

    corresponding_axes.set(title=axis_title)
    figure.colorbar(spectrogram, ax=corresponding_axes, format="%+2.f dB")


def weiner_filter_denoising(audio_file_path: pathlib.Path, output_dir: pathlib.Path):
    audio_data, sr = librosa.load(audio_file_path, sr=None)

    filtered_audio = denoise.apply_iterative_wiener(
        audio_data, frame_len=512, lpc_order=20, iterations=2, alpha=0.8, thresh=0.01
    )

    # peak normalize audio to -1 dB
    peak_normalized_audio = pyln.normalize.peak(filtered_audio, -12.0)

    # measure the loudness first
    meter = pyln.Meter(sr)  # create BS.1770 meter
    loudness = meter.integrated_loudness(filtered_audio)

    # loudness normalize audio to -6 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(filtered_audio, loudness, -12.0)

    output_path = pathlib.Path(output_dir, f"{audio_file_path.stem}_weiner.wav")
    sf.write(output_path, peak_normalized_audio, sr, subtype="FLOAT")

    return output_path


def samplerate_preprocess(
    input_file: pathlib.Path, desired_samplerate: int
) -> pathlib.Path:
    data, input_sampleraete = librosa.load(input_file, sr=None)

    if input_sampleraete == desired_samplerate:
        return input_file

    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        resampled_data = librosa.resample(
            data, orig_sr=input_sampleraete, target_sr=desired_samplerate
        )
        output_path = pathlib.Path(temp_dir, input_file.name)
        sf.write(output_path, resampled_data.T, desired_samplerate)
        return output_path


def rnnoised_amalgamated_denoising(
    rnnoise_binary: pathlib.Path, file_path: pathlib.Path, output_dir: pathlib.Path
):
    input_file_path = file_path
    if file_path.suffix == ".wav":
        input_file_path = samplerate_preprocess(file_path, 48000)

    execution_cmd = []
    execution_cmd.append(rnnoise_binary)
    execution_cmd.append(f"--input={input_file_path}")
    result_filename = f"{input_file_path.stem}_rnnoise_denoising.wav"
    result_path = pathlib.Path(output_dir, result_filename)
    execution_cmd.append(f"--output={result_path}")
    _result = subprocess.run(
        execution_cmd, check=True, capture_output=True, cwd=output_dir
    )

    return result_path


# def process_with_agc(input_file: pathlib.Path):
#     # read audiofile
#     sr, d = scipy.io.wavfile.read(input_file)

#     # convert from int16 to float (-1,1) range
#     convert_16_bit = float(2**15)
#     d = d / (convert_16_bit + 1.0)

#     # apply AGC
#     (y, D, E) = tf_agc(d, sr, plot=True)

#     # convert back to int16 to save
#     y = np.int16(y / np.max(np.abs(y)) * convert_16_bit)
#     output_result = f"{input_file.stem}_agc_result.wav"
#     scipy.io.wavfile.write(output_result, sr, y)


def main():
    parser = argparse.ArgumentParser(
        description="Launch rnnoise test app for processing"
    )
    parser.add_argument(
        "--rnnoise-binary",
        type=str,
        help="Path to the rnnoise compiled binary.",
        default=str(
            pathlib.Path(os.getcwd(), "build", "examples", "rnnoise_libsoundfile")
        ),
    )
    parser.add_argument(
        "--input-directory",
        type=str,
        help="Path to the directory containing noisy audio files.",
        default=str(pathlib.Path(os.getcwd(), "sample_audios")),
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        help="Path to the directory where to store the results.",
        default=str(pathlib.Path(os.getcwd(), "evaluation_output")),
    )

    args = parser.parse_args()
    binary_path = args.rnnoise_binary
    directory_path = args.input_directory
    output_dir = args.output_directory

    if not os.path.isfile(binary_path):
        print(f"Error: The binary file '{binary_path}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        sys.exit(1)

    if pathlib.Path(output_dir).exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    sigmos_evaluator = SigmosModelRunner()

    for filename in os.listdir(directory_path):
        input_audio_file_path = pathlib.Path(directory_path, filename)

        if os.path.isfile(input_audio_file_path):
            try:
                print(f"Processing file: {input_audio_file_path}")

                denoised_by_rnnoise = rnnoised_amalgamated_denoising(
                    binary_path, input_audio_file_path, output_dir
                )
                denoised_by_weiner = weiner_filter_denoising(
                    input_audio_file_path, output_dir
                )

                fig, (source_axes, denoised_weiner_axes, denoised_rnnoise_axes) = (
                    plt.subplots(3, 1, figsize=(12, 12))
                )

                plot_spectrogram(
                    input_audio_file_path,
                    source_axes,
                    fig,
                    f"Source signal:{input_audio_file_path.stem}",
                )

                plot_spectrogram(
                    denoised_by_weiner,
                    denoised_weiner_axes,
                    fig,
                    f"Adaptive Weiner filter:{input_audio_file_path.stem}",
                )

                plot_spectrogram(
                    denoised_by_rnnoise,
                    denoised_rnnoise_axes,
                    fig,
                    f"Amalgamated RNNOISE neural network: {input_audio_file_path.stem}",
                )
                plt.tight_layout()

                output_path_picture = pathlib.Path(
                    output_dir, f"{input_audio_file_path.stem}_spectrogram.png"
                )

                plt.savefig(output_path_picture)
                plt.close(fig)

                input_file_result = sigmos_evaluator.evaluate_file(
                    input_audio_file_path
                )
                weiner_filter_result = sigmos_evaluator.evaluate_file(
                    denoised_by_weiner
                )
                rnnoise_denoising_result = sigmos_evaluator.evaluate_file(
                    denoised_by_rnnoise
                )

                plot_sigmos_results(
                    sigmos_items = [
                        SigmosResultItem(input_file_result, input_audio_file_path),
                        SigmosResultItem(weiner_filter_result, denoised_by_weiner),
                        SigmosResultItem(rnnoise_denoising_result, denoised_by_rnnoise),
                    ],
                    output_path = pathlib.Path(
                        output_dir, f"{input_audio_file_path.stem}_sigmos.png"
                    ),
                    evaluation_title=f"{input_audio_file_path.stem}"
                )

            except subprocess.CalledProcessError as e:
                print(f"Error occured during processing: {input_audio_file_path}")
                print(e.stderr.decode())


if __name__ == "__main__":
    main()
