#include "sndfile.hh"
#include "cxxopts.hpp"

#include "rnnoise.h"
#include <cstdint>
#include <filesystem>
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <array>
#include <memory>
#include <utility>

template <auto DeleterFunction>
using CustomDeleter = std::integral_constant<decltype(DeleterFunction), DeleterFunction>;

template <typename ManagedType, auto Functor>
using PointerWrapper = std::unique_ptr<ManagedType, CustomDeleter<Functor>>;


inline constexpr std::size_t AUDIO_BUFFER_LENGTH = 480;
inline constexpr std::size_t NUM_CHANNELS = 1;
inline constexpr std::size_t SAMPLERATE = 48000;

using RNNoiseDenoiseStatePtr = PointerWrapper<DenoiseState,rnnoise_destroy>;
using RnnModelPtr = PointerWrapper<RNNModel,rnnoise_model_free>;

RnnModelPtr rnn_model_ptr;
RNNoiseDenoiseStatePtr rnnoise_denoise_state_ptr;

static void initialize_rnnoise_library(){
    rnnoise_denoise_state_ptr.reset(rnnoise_create(nullptr));
}
void process_audio_recording(const std::filesystem::path& input_file, const std::filesystem::path& output_file){
    SndfileHandle input_audio_file_handle{SndfileHandle(input_file.c_str())};

    spdlog::info("Opened input audio file:{}", input_file.c_str());
    spdlog::info("Number of channels:{}", input_audio_file_handle.channels());
    spdlog::info("Samplerate:{}", input_audio_file_handle.samplerate());

    SndfileHandle output_audio_file_handle{SndfileHandle{
        output_file.c_str(),
        SFM_WRITE,
        SF_FORMAT_WAV | SF_FORMAT_PCM_16,
        NUM_CHANNELS,
        SAMPLERATE
        }
    };

    using TSamplesBufferArray = std::array<float,AUDIO_BUFFER_LENGTH>;
    static TSamplesBufferArray samples_buffer{};

    spdlog::info("Processing audio...");
    while (input_audio_file_handle.read (samples_buffer.data(), samples_buffer.size()) != 0) {
        float vad_confidence = rnnoise_process_frame(rnnoise_denoise_state_ptr.get(), samples_buffer.data(), samples_buffer.data());
        spdlog::info("VAD confidence is:{}",vad_confidence);

        output_audio_file_handle.write(samples_buffer.data(),samples_buffer.size());
    }
    spdlog::info("Processing done. WAVE file can be found at: {}", output_file.c_str());
}

int main(int argc, char** argv){
    cxxopts::Options options("rnnoise_libsoundfile denoiser", "Simple runner of rnnoise over WAVe files with 48K samplerate");
    options.add_options()
    ("input", "Input file to process",cxxopts::value<std::filesystem::path>())
    ("output", "Output file", cxxopts::value<std::filesystem::path>())
     ("help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        fmt::print(options.help());
        exit(0);
    }

    const auto input_file_path_opt = result["input"].as<std::filesystem::path>();
    const auto output_file_path_opt = result["output"].as<std::filesystem::path>();

    initialize_rnnoise_library();
    process_audio_recording(input_file_path_opt,output_file_path_opt);
    return 0;
}