# ./amalgamation/Heady_amalgate -s src -e "config.h compile.sh rnn_train.py write_weights.c rnnoise_data.c vec_avx.h"  -o "header_only/rnnoise_amalgamated.c" -d RNNOISE_ALMAGAMATED

import pathlib
import subprocess
import os
import argparse
from dataclasses import dataclass

@dataclass
class HeadyPackage:
    source_directory:pathlib.Path

    def amalgamate_sources(self, rnnoise_sources_path:pathlib.Path):
        self._build_if_necessary()

        command = [
            f"{pathlib.Path(self.build_directory,'Heady')}",
            "-s",
            "src",
            "-e",
            "config.h compile.sh rnn_train.py write_weights.c rnnoise_data.c vec_avx.h",
            "-o",
            "header_only/rnnoise_amalgamated.c",
            "-d",
            "RNNOISE_ALMAGAMATED",
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=rnnoise_sources_path
            )
            print("Command executed successfully.")
            print("Output:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error occurred while executing the command.")
            print("Error message:\n", e.stderr)


    def _build_if_necessary(self):
        self.build_directory = pathlib.Path(self.source_directory,"build")
        cmake_configure = [
            "cmake",
            "-G",
            "Unix Makefiles",
            f"-S={self.source_directory}",
            f"-B={self.build_directory}",
        ]
        subprocess.run(
            cmake_configure,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        cmake_build = [
            "cmake",
            "--build",
            f"{self.build_directory}",
        ]

        subprocess.run(
            cmake_build,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Rnnoise amalgamation tool based on Heady package")
    
    parser.add_argument("--heady-source", type=pathlib.Path, help="Source directory of Heady package")
    parser.add_argument("--rnnoise-source", type=pathlib.Path, help="Source directory with rnnoise sources")
    
    args = parser.parse_args()
    
    heady_amalgamation_tool = HeadyPackage(source_directory=args.heady_source)
    heady_amalgamation_tool.amalgamate_sources(args.rnnoise_source)

if __name__ == "__main__":
    main()
