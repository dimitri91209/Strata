// include/strata/flydsl_nvfp4/engine.hpp - Bolt-on FlyDSL NVFP4 engine for Qwen3.8-Flash-Next (qwen4_exp) on gfx120x.
#pragma once

namespace strata::flydsl_nvfp4 {

/// Runs the Qwen3.8-Flash-Next NVFP4 safetensors model on AMD RDNA4 (gfx1201) routing all GPU compute through
/// FlyDSL kernels (JIT-compiled on demand and cached on disk). Activated when `--flydsl-nvfp4` is passed to `strata`.
int run(int argc, char** argv);

}  // namespace strata::flydsl_nvfp4
