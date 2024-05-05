#include "sndfile.hh"
#include "cxxopts.hpp"

#include "rnnoise.h"
#include <cstdint>
#include <filesystem>
#include <fmt/core.h>

inline constexpr std::size_t kAudioBufferLength = 480;


int main(int argc, char** argv){
    cxxopts::Options options("rnnoise_libsoundfile denoiser", "Simple runner of rnnoise over WAVe files with 48K samplerate");
    options.add_options()
    ("i,input-file", "Input file to process",cxxopts::value<std::filesystem::path>())
    ("o,output", "Output file", cxxopts::value<std::filesystem::path>())
     ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        fmt::print(options.help());
        exit(0);
    }

    return 0;
}