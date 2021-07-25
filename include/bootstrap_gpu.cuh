/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "details/allocator_gpu.cuh"
#include "cufhe_gpu.cuh"
#include "encoder.cuh"

#include <params.hpp>

namespace cufhe {
void InitializeNTThandlers(const int gpuNum);
void BootstrappingKeyToNTT(
    const TFHEpp::BootstrappingKey<TFHEpp::lvl01param>& bk, const int gpuNum);
void KeySwitchingKeyToDevice(
    const TFHEpp::KeySwitchingKey<TFHEpp::lvl10param>& ksk, const int gpuNum);
void DeleteBootstrappingKeyNTT(const int gpuNum);
void DeleteKeySwitchingKey(const int gpuNum);
void CMUXNTTkernel(TFHEpp::lvl1param::T* res, const FFP* const cs, TFHEpp::lvl1param::T* const c1, TFHEpp::lvl1param::T* const c0,
                         cudaStream_t st, const int gpuNum);
void Bootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in,
               TFHEpp::lvl1param::T mu, cudaStream_t st, const int gpuNum);
void BootstrapTLWE2TRLWE(TFHEpp::lvl1param::T* out, TFHEpp::lvl0param::T* in,
                         TFHEpp::lvl1param::T mu, cudaStream_t st, const int gpuNum);
void ProgrammableBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in,
    cudaStream_t st, const int gpuNum, EncoderDevice *encoder_domain, EncoderDevice *encoder_target, double (*function)(double));
void SEIandBootstrap2TRLWE(TFHEpp::lvl1param::T* out, TFHEpp::lvl1param::T* in,
                         lvl1param::T mu, cudaStream_t st, const int gpuNum);
void SEandKS(TFHEpp::lvl0param::T* out, TFHEpp::lvl1param::T* in,
             cudaStream_t st, const int gpuNum);

void NandBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                   TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void OrBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                 TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void OrYNBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                   TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void OrNYBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                   TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void AndBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                  TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void AndYNBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                    TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void AndNYBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                    TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void NorBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                  TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void XorBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                  TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void XnorBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in0,
                   TFHEpp::lvl0param::T* in1, cudaStream_t st, const int gpuNum);
void CopyBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in,
                   cudaStream_t st, const int gpuNum);
void NotBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* in,
                  cudaStream_t st, const int gpuNum);
void MuxBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* inc,
                  TFHEpp::lvl0param::T* in1, TFHEpp::lvl0param::T* in0,
                  cudaStream_t st, const int gpuNum);
void NMuxBootstrap(TFHEpp::lvl0param::T* out, TFHEpp::lvl0param::T* inc,
                  TFHEpp::lvl0param::T* in1, TFHEpp::lvl0param::T* in0,
                  cudaStream_t st, const int gpuNum);
}  // namespace cufhe
