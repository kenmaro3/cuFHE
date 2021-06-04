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

/**
 * @file cufhe.h
 * @brief This is the user API of the cuFHE library.
 *        It hides most of the contents in the developer API and
 *        only provides essential data structures and functions.
 */

#pragma once
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <array>

#include "../thirdparties/TFHEpp/include/cloudkey.hpp"
#include "cufhe.h"

namespace cufhe {

extern int _gpuNum;

extern int streamCount;

/**
 * Call before running gates on server.
 * 1. Generate necessary NTT data.
 * 2. Convert BootstrappingKey to NTT form.
 * 3. Copy KeySwitchingKey to GPU memory.
 */
void SetGPUNum(int gpuNum);

// Initialize NTThandlers only.
void Initialize();

void Initialize(const TFHEpp::GateKeywoFFT& gk);

/** Remove everything created in Initialize(). */
void CleanUp();

/**
 * \brief Synchronize device.
 * \details This makes it easy to wrap in python.
 */
inline void Synchronize()
{
    for (int i = 0; i < _gpuNum; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
};

/**
 * \class Stream
 * \brief This is created for easier wrapping in python.
 */
class Stream {
   public:
    inline Stream()
    {
        st_ = 0;
        _device_id = streamCount % _gpuNum;
        streamCount++;
    }
    inline Stream(int device_id)
    {
        _device_id = device_id;
        st_ = 0;
        streamCount++;
    }

    inline ~Stream()
    {
        // Destroy();
    }

    inline void Create()
    {
        cudaSetDevice(_device_id);
        cudaStreamCreateWithFlags(&this->st_, cudaStreamNonBlocking);
    }

    inline void Destroy()
    {
        cudaSetDevice(_device_id);
        cudaStreamDestroy(this->st_);
    }
    inline cudaStream_t st() { return st_; };
    inline int device_id() { return _device_id; }

   private:
    cudaStream_t st_;
    int _device_id;
};  // class Stream

void TRGSW2NTT(cuFHETRGSWNTTlvl1& trgswntt, const TFHEpp::TRGSW<TFHEpp::lvl1param>& trgsw, Stream st);
void GateBootstrappingTLWE2TRLWElvl01NTT(cuFHETRLWElvl1& out, Ctxt& in,
                                         Stream st);
void SampleExtractAndKeySwitch(Ctxt& out, const cuFHETRLWElvl1& in, Stream st);
void CMUXNTT(cuFHETRLWElvl1& res, cuFHETRGSWNTTlvl1& cs, cuFHETRLWElvl1& c1, cuFHETRLWElvl1& c0, Stream st);
void And(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void AndYN(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void AndNY(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void Or(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void OrYN(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void OrNY(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void Nand(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void Nor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void Xor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void Xnor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void Not(Ctxt& out, Ctxt& in, Stream st);
void Copy(Ctxt& out, Ctxt& in, Stream st);
void CopyOnHost(Ctxt& out, Ctxt&);
void Mux(Ctxt& out, Ctxt& inc, Ctxt& in1, Ctxt& in0, Stream st);

bool StreamQuery(Stream st);
void CtxtCopyH2D(Ctxt& c, Stream st);
void CtxtCopyD2H(Ctxt& c, Stream st);
void TRLWElvl1CopyH2D(cuFHETRLWElvl1& c, Stream st);
void TRLWElvl1CopyD2H(cuFHETRLWElvl1& c, Stream st);
void TRGSWNTTlvl1CopyH2D(cuFHETRGSWNTTlvl1& c, Stream st);

void gSampleExtractAndKeySwitch(Ctxt& out, const cuFHETRLWElvl1& in, Stream st);
void gGateBootstrappingTLWE2TRLWElvl01NTT(cuFHETRLWElvl1& out, Ctxt& in,
                                          Stream st);
void gNand(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gOr(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gOrYN(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gOrNY(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gAnd(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gAndYN(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gAndNY(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gNor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gXor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gXnor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st);
void gNot(Ctxt& out, Ctxt& in, Stream st);
void gMux(Ctxt& out, Ctxt& inc, Ctxt& in1, Ctxt& in0, Stream st);
void gCopy(Ctxt& out, Ctxt& in, Stream st);
void gCMUXNTT(cuFHETRLWElvl1& res, cuFHETRGSWNTTlvl1& cs, cuFHETRLWElvl1& c1, cuFHETRLWElvl1& c0, Stream st);

}  // namespace cufhe
