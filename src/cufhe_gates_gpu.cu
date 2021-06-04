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

#include <include/cufhe.h>
#include <unistd.h>

#include <array>
#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>

#include "../thirdparties/TFHEpp/include/cloudkey.hpp"
#include "../thirdparties/TFHEpp/include/params.hpp"

namespace cufhe {

int _gpuNum = 1;

int streamCount = 0;

void SetGPUNum(int gpuNum) { _gpuNum = gpuNum; }

void Initialize()
{
    InitializeNTThandlers(_gpuNum);
}

void Initialize(const TFHEpp::GateKeywoFFT& gk)
{
    InitializeNTThandlers(_gpuNum);
    BootstrappingKeyToNTT(gk.bklvl01, _gpuNum);
    KeySwitchingKeyToDevice(gk.ksk, _gpuNum);
}

void CleanUp()
{
    DeleteBootstrappingKeyNTT(_gpuNum);
    DeleteKeySwitchingKey(_gpuNum);
}

inline void CtxtCopyH2D(Ctxt& c, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(c.tlwedevices[st.device_id()], c.tlwehost.data(),
                    sizeof(c.tlwehost), cudaMemcpyHostToDevice, st.st());
}

inline void CtxtCopyD2H(Ctxt& c, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(c.tlwehost.data(), c.tlwedevices[st.device_id()],
                    sizeof(c.tlwehost), cudaMemcpyDeviceToHost, st.st());
}

void TRLWElvl1CopyH2D(cuFHETRLWElvl1& c, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(c.trlwedevices[st.device_id()], c.trlwehost.data(),
                    sizeof(c.trlwehost), cudaMemcpyHostToDevice, st.st());
}

void TRLWElvl1CopyD2H(cuFHETRLWElvl1& c, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(c.trlwehost.data(), c.trlwedevices[st.device_id()],
                    sizeof(c.trlwehost), cudaMemcpyDeviceToHost, st.st());
}

void TRGSWNTTlvl1CopyH2D(cuFHETRGSWNTTlvl1& c, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(c.trgswdevices[st.device_id()], c.trgswhost.data(),
                    sizeof(c.trgswhost), cudaMemcpyHostToDevice, st.st());
}

void CMUXNTT(cuFHETRLWElvl1& res, cuFHETRGSWNTTlvl1& cs, cuFHETRLWElvl1& c1, cuFHETRLWElvl1& c0,
                                         Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(cs.trgswdevices[st.device_id()], cs.trgswhost.data(),
                    sizeof(cs.trgswhost), cudaMemcpyHostToDevice, st.st());
    cudaMemcpyAsync(c1.trlwedevices[st.device_id()], c1.trlwehost.data(),
                    sizeof(c1.trlwehost), cudaMemcpyHostToDevice, st.st());
    cudaMemcpyAsync(c0.trlwedevices[st.device_id()], c0.trlwehost.data(),
                    sizeof(c0.trlwehost), cudaMemcpyHostToDevice, st.st());
    CMUXNTTkernel(res.trlwedevices[st.device_id()], cs.trgswdevices[st.device_id()],
                        c1.trlwedevices[st.device_id()], c0.trlwedevices[st.device_id()], st.st(),
                        st.device_id());
    cudaMemcpyAsync(res.trlwehost.data(), res.trlwedevices[st.device_id()],
                    sizeof(res.trlwehost), cudaMemcpyDeviceToHost, st.st());
}

void gCMUXNTT(cuFHETRLWElvl1& res, cuFHETRGSWNTTlvl1& cs, cuFHETRLWElvl1& c1, cuFHETRLWElvl1& c0, Stream st)
{
    cudaSetDevice(st.device_id());
    CMUXNTTkernel(res.trlwedevices[st.device_id()], cs.trgswdevices[st.device_id()],
                  c1.trlwedevices[st.device_id()], c0.trlwedevices[st.device_id()], st.st(),
                  st.device_id());
}

void GateBootstrappingTLWE2TRLWElvl01NTT(cuFHETRLWElvl1& out, Ctxt& in,
                                         Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in, st);
    BootstrapTLWE2TRLWE(out.trlwedevices[st.device_id()],
                        in.tlwedevices[st.device_id()], 1U << 29, st.st(),
                        st.device_id());
    cudaMemcpyAsync(out.trlwehost.data(), out.trlwedevices[st.device_id()],
                    sizeof(out.trlwehost), cudaMemcpyDeviceToHost, st.st());
}

void gGateBootstrappingTLWE2TRLWElvl01NTT(cuFHETRLWElvl1& out, Ctxt& in,
                                          Stream st)
{
    cudaSetDevice(st.device_id());
    BootstrapTLWE2TRLWE(out.trlwedevices[st.device_id()],
                        in.tlwedevices[st.device_id()], 1U << 29, st.st(),
                        st.device_id());
    cudaMemcpyAsync(out.trlwehost.data(), out.trlwedevices[st.device_id()],
                    sizeof(out.trlwehost), cudaMemcpyDeviceToHost, st.st());
}

void SampleExtractAndKeySwitch(Ctxt& out, const cuFHETRLWElvl1& in, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(in.trlwedevices[st.device_id()], in.trlwehost.data(),
                    sizeof(in.trlwehost), cudaMemcpyHostToDevice, st.st());
    SEandKS(out.tlwedevices[st.device_id()], in.trlwedevices[st.device_id()],
            st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gSampleExtractAndKeySwitch(Ctxt& out, const cuFHETRLWElvl1& in, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(in.trlwedevices[st.device_id()], in.trlwehost.data(),
                    sizeof(in.trlwehost), cudaMemcpyHostToDevice, st.st());
    SEandKS(out.tlwedevices[st.device_id()], in.trlwedevices[st.device_id()],
            st.st(), st.device_id());
}

void Nand(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    NandBootstrap(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gNand(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    NandBootstrap(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void Or(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    OrBootstrap(out.tlwedevices[st.device_id()],
                in0.tlwedevices[st.device_id()],
                in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gOr(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrBootstrap(out.tlwedevices[st.device_id()],
                in0.tlwedevices[st.device_id()],
                in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void OrYN(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    OrYNBootstrap(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gOrYN(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrYNBootstrap(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void OrNY(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    OrNYBootstrap(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gOrNY(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrNYBootstrap(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void And(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    AndBootstrap(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gAnd(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndBootstrap(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void AndYN(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    AndYNBootstrap(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gAndYN(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndYNBootstrap(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void AndNY(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    AndNYBootstrap(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gAndNY(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndNYBootstrap(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void Nor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    NorBootstrap(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gNor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    NorBootstrap(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void Xor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    XorBootstrap(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gXor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    XorBootstrap(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void Xnor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    XnorBootstrap(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gXnor(Ctxt& out, Ctxt& in0, Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    XnorBootstrap(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void Not(Ctxt& out, Ctxt& in, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in, st);
    NotBootstrap(out.tlwedevices[st.device_id()],
                 in.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gNot(Ctxt& out, Ctxt& in, Stream st)
{
    cudaSetDevice(st.device_id());
    NotBootstrap(out.tlwedevices[st.device_id()],
                 in.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void Copy(Ctxt& out, Ctxt& in, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in, st);
    CopyBootstrap(out.tlwedevices[st.device_id()],
                  in.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gCopy(Ctxt& out, Ctxt& in, Stream st)
{
    cudaSetDevice(st.device_id());
    CopyBootstrap(out.tlwedevices[st.device_id()],
                  in.tlwedevices[st.device_id()], st.st(), st.device_id());
}

void CopyOnHost(Ctxt& out, Ctxt& in) { out.tlwehost = in.tlwehost; }

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
void Mux(Ctxt& out, Ctxt& inc, Ctxt& in1, Ctxt& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(inc, st);
    CtxtCopyH2D(in1, st);
    CtxtCopyH2D(in0, st);
    MuxBootstrap(out.tlwedevices[st.device_id()],
                 inc.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gMux(Ctxt& out, Ctxt& inc, Ctxt& in1, Ctxt& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    MuxBootstrap(out.tlwedevices[st.device_id()],
                 inc.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()], st.st(), st.device_id());
}

// void SetToGPU(Ctxt& in)
// {
//     cudaMemcpy(in.lwe_sample_device_->data(), in.lwe_sample_->data(),
//                in.lwe_sample_->SizeData(), cudaMemcpyHostToDevice);
// }

// void GetFromGPU(Ctxt& out)
// {
//     cudaMemcpy(out.lwe_sample_->data(), out.lwe_sample_device_->data(),
//                out.lwe_sample_->SizeData(), cudaMemcpyDeviceToHost);
// }

bool StreamQuery(Stream st)
{
    cudaSetDevice(st.device_id());
    cudaError_t res = cudaStreamQuery(st.st());
    if (res == cudaSuccess) {
        return true;
    }
    else {
        return false;
    }
}
}  // namespace cufhe
