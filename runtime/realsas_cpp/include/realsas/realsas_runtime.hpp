#ifndef REALSAS_RUNTIME_HPP
#define REALSAS_RUNTIME_HPP

#include "realsas_runtime.h"
#include <stdexcept>
#include <string>
#include <utility>

namespace realsas {

class Runtime {
public:
    explicit Runtime(const std::string& path) {
        char error[1024] = {};
        const RsResult result = rs_runtime_open(path.c_str(), &handle_, error, sizeof(error));
        if (result != RS_OK) {
            throw std::runtime_error(error[0] ? error : "Failed to open RealSaS runtime package");
        }
    }
    ~Runtime() { rs_runtime_close(handle_); }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Runtime& operator=(Runtime&& other) noexcept {
        if (this != &other) {
            rs_runtime_close(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    RsRuntime* get() noexcept { return handle_; }
    const RsRuntime* get() const noexcept { return handle_; }
private:
    RsRuntime* handle_ = nullptr;
};

} // namespace realsas
#endif
