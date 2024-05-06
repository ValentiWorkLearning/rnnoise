add_executable(rnnoise_demo rnnoise_demo.c)

target_link_libraries(rnnoise_demo PRIVATE rnnoise)

find_package(SndFile REQUIRED)
find_package(cxxopts REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

add_executable(rnnoise_libsoundfile rnnoise_libsndfile.cpp)
target_link_libraries(
    rnnoise_libsoundfile
    PRIVATE
    rnnoise
    SndFile::sndfile
    cxxopts::cxxopts
    fmt::fmt
    spdlog::spdlog
)
