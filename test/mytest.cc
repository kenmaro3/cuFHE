// Include these two files for GPU computing.
#include <include/cufhe_gpu.cuh>
#include <include/encoder.cuh>
#include <include/bootstrap_gpu.cuh>
#include "plain.h"
#include <test/test_util.h>
using namespace cufhe;

#include <iostream>
#include <random>
#include <vector>
using namespace std;

void mytest(){
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const uint32_t kNumSMs = prop.multiProcessorCount;
    const uint32_t kNumTests = kNumSMs * 32;   // * 8;
    constexpr uint32_t kNumLevels = 10;  // Gate Types, Mux is counted as 2.

    double encoder_a = -20.;
    double encoder_b = 20.;
    int bs_bp = 32;

    cufhe::EncoderDevice* encoder_device = new EncoderDevice(encoder_a, encoder_b, bs_bp);
    TFHEpp::Encoder encoder_host(encoder_a, encoder_b, bs_bp);

    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    //TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk, encoder_host);
    TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk);
    TFHEpp::GateKey* gk2 = new TFHEpp::GateKey(*sk);

    cout << "n:" << sk->params.lvl0.n << endl;

    double x = -15.0;

    for(int i=0; i<3; i++){
        Ctxt c1, c2;
        c1.tlwehost = TFHEpp::tlweSymEncodeEncrypt<lvl0param>(x, lvl0param::alpha, sk->key.lvl0, encoder_host);
        double d = TFHEpp::tlweSymDecryptDecode<lvl0param>(c1.tlwehost, sk->key.lvl0, encoder_host);
        printf("before anything: %f\n", d);


        Synchronize();

        Initialize(*gk);  // essential for GPU computing

        Stream* st = new Stream;
        st->Create();

        float et;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaSetDevice(st->device_id());
        CtxtCopyH2D(c1, *st);
        //CtxtCopyH2D(in1, st);
        ProgrammableBootstrap(c2.tlwedevices[st->device_id()], c1.tlwedevices[st->device_id()], st->st(), st->device_id(),
                            encoder_device, encoder_device, my_identity_function_device);
        Bootstrap(c2.tlwedevices[st->device_id()], c1.tlwedevices[st->device_id()],
                lvl1param::mu, st->st(), st->device_id());
        //ProgrammableBootstrapWithoutKS(c3.tlwedevices[st->device_id()], c1.tlwedevices[st->device_id()], st->st(), st->device_id(),
        //                    encoder_device, encoder_device, my_identity_function_device);
        CtxtCopyD2H(c2, *st);

        Synchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&et, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        double dec = TFHEpp::tlweSymDecryptDecode<lvl0param>(c2.tlwehost, sk->key.lvl0, encoder_host);
        printf("res: %f\n", dec);
        st->Destroy();
        delete st;
        CleanUp();  // essential to clean and deallocate data

    }
}


void test_nand(){
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    //TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk, encoder_host);
    TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk);

    cout << "n:" << sk->params.lvl0.n << endl;

    for(int i=0; i<3; i++){
        Ctxt c1, c2, c3;
        c1.tlwehost = TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(
            TFHEpp::lvl0param::mu,TFHEpp::lvl0param::alpha, sk->key.lvl0);
        c2.tlwehost = TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(
            -TFHEpp::lvl0param::mu,TFHEpp::lvl0param::alpha, sk->key.lvl0);

        Synchronize();
        Initialize(*gk);  // essential for GPU computing

        Stream* st = new Stream;
        st->Create();

        float et;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaSetDevice(st->device_id());
        CtxtCopyH2D(c1, *st);
        CtxtCopyH2D(c2, *st);
        //NandBootstrap(c3.tlwedevices[st->device_id()],
        //            c1.tlwedevices[st->device_id()],
        //            c2.tlwedevices[st->device_id()], st->st(), st->device_id());
        Bootstrap(c3.tlwedevices[st->device_id()], c1.tlwedevices[st->device_id()],
               lvl1param::mu, st->st(), st->device_id());
        CtxtCopyD2H(c3, *st);
        uint8_t res;
        res = TFHEpp::tlweSymDecrypt<TFHEpp::lvl0param>(c3.tlwehost, sk->key.lvl0);
        printf("res: %d\n", res);
                                                        

        Synchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&et, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        st->Destroy();
        delete st;
        CleanUp();  // essential to clean and deallocate data

    }
}

int main()
{
    //test_nand();
    mytest();
}
