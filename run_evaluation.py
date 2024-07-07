import os
import sys
import argparse
import subprocess
import pathlib
import tempfile

import soundfile as sf
import librosa


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft


def plot_spectrogram(file_path:pathlib.Path, ax, title:str):
    sample_rate, samples = wavfile.read(file_path)
    frequencies, times, Zxx = stft(samples, sample_rate)
    ax.pcolormesh(times, frequencies, 10 * np.log10(np.abs(Zxx)**2), shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(title)
    ax.set_ylim(0, sample_rate / 2)

def samplerate_preprocess(input_file:pathlib.Path, desired_samplerate:int)->pathlib.Path:
    data, input_sampleraete = librosa.load(input_file, sr=None)
    
    if input_sampleraete == desired_samplerate:
        return input_file


    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        resampled_data = librosa.resample(data, orig_sr=input_sampleraete, target_sr=desired_samplerate)
        output_path = pathlib.Path(temp_dir,input_file.name)
        sf.write(output_path, resampled_data.T, desired_samplerate)
        return output_path
        

def main():

    parser = argparse.ArgumentParser(description="Launch rnnoise test app for processing")
    parser.add_argument('--rnnoise-binary', type=str, help='Path to the rnnoise compiled binary.')
    parser.add_argument('--input-directory', type=str, help='Path to the directory containing noisy audio files.')
    parser.add_argument('--output-directory', type=str, help='Path to the directory where to store the results.')

    args = parser.parse_args()
    binary_path = args.rnnoise_binary
    directory_path = args.input_directory
    output_path = args.output_directory

    if not os.path.isfile(binary_path):
        print(f"Error: The binary file '{binary_path}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(directory_path):
        file_path = pathlib.Path(directory_path, filename)

        if os.path.isfile(file_path):
            try:
                print(f"Processing file: {file_path}")
                input_file_path = file_path
                if(file_path.suffix == ".wav"):
                    input_file_path = samplerate_preprocess(file_path,48000)

                execution_cmd = []
                execution_cmd.append(binary_path)
                execution_cmd.append(f"--input={input_file_path}")
                result_filename = f"{input_file_path.stem}_denoised.wav"
                result_path = pathlib.Path(output_path,result_filename)
                execution_cmd.append(f"--output={result_path}")
                result = subprocess.run(execution_cmd, check=True, capture_output=True, cwd=output_path)

                
                fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
                plot_spectrogram(input_file_path, axs[0], 'Spectrogram of ' + pathlib.Path(input_file_path).stem)
                plot_spectrogram(result_path, axs[1], 'Spectrogram of ' + pathlib.Path(result_path).stem)
                plt.tight_layout()
                
                output_path_picture = os.path.join(output_path, f'{pathlib.Path(input_file_path).stem}_spectrogram.png')
                plt.savefig(output_path_picture)
                plt.close(fig)
 
            except subprocess.CalledProcessError as e:
                print(f"Error processing {file_path}")
                print(e.stderr.decode())

if __name__ == "__main__":
    main()