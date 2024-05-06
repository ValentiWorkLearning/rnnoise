#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>

class LazyFileWriter {
private:
    std::filesystem::path m_filepath;
    std::fstream fileStream;
    bool m_is_open;
    void openFileIfNeeded() {
        if (!m_is_open) {
            fileStream.open(m_filepath, std::ios::out | std::ios::app);
            if (!fileStream.is_open()) {
                spdlog::error("Failed to open file:{}",m_filepath.c_str());
                return;
            }
            m_is_open = true;
        }
    }

public:
    LazyFileWriter(const std::filesystem::path& filepath) : m_filepath{filepath}, m_is_open{false} {}

    ~LazyFileWriter() {
        if (m_is_open) {
            fileStream.close();
        }
    }

    template<typename TypeToWrite>
    void write(TypeToWrite&& data) {
        openFileIfNeeded();
        if (m_is_open) {
            fileStream << data << std::endl;
        }
    }
};