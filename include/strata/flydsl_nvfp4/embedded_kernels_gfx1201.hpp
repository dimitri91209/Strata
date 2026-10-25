// Auto-generated embedded gfx1201 HSACO kernel table for Strata-gfx1201-nvfp4-native.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace strata::flydsl_nvfp4 {

struct EmbeddedKernelDesc {
    const char* key;
    const char* kernel_name;
    int bx, by, bz;
    int gx, gy, gz;
    const uint8_t* hsaco_data;
    size_t hsaco_size;
};

const EmbeddedKernelDesc* find_embedded_kernel(std::string_view key);
size_t embedded_kernel_count();
const EmbeddedKernelDesc* embedded_kernels_begin();

}  // namespace strata::flydsl_nvfp4
