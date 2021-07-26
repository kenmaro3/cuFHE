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

#include <chrono>
using namespace std::chrono;
inline double get_time_msec(void){
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count())/1000000;
}

void test_nand(){
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    //TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk, encoder_host);
    TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk);

    cout << "n:" << sk->params.lvl0.n << endl;

    for(int i=0; i<1; i++){
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

    cufhe::EncoderDevice* encoder_device;
    TFHEpp::Encoder encoder_host(encoder_a, encoder_b, bs_bp);
    cufhe::EncoderDevice* encoder = new EncoderDevice(encoder_a, encoder_b, bs_bp);

    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk);
    TFHEpp::GateKey* gk2 = new TFHEpp::GateKey(*sk);

    cout << "n:" << sk->params.lvl0.n << endl;


    vector<double> ts;
    double tstart, tend;
    for(int i=1; i<2; i++){
        Ctxt c1, c2;
        //double x = encoder_a + double(i)*encoder_host.half_d/11.;
        double x = 5.;
        c1.tlwehost = TFHEpp::tlweSymEncodeEncrypt<lvl0param>(x, lvl0param::alpha, sk->key.lvl0, encoder_host);
        double d = TFHEpp::tlweSymDecryptDecode<lvl0param>(c1.tlwehost, sk->key.lvl0, encoder_host);

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

        // ##############################################
        encoder->mult_number = 0.5;
        cudaMalloc((void **)&encoder_device, sizeof(EncoderDevice));
        cudaMemcpy(encoder_device, encoder, sizeof(EncoderDevice), cudaMemcpyHostToDevice);

        // function_type -> {0: identity, 1: relu, 2: sigmoid, 3: mult}
        int function_type = 3;
        int * function_type_d;
        cudaMalloc((void **)&function_type_d, sizeof(int));
        cudaMemcpy(function_type_d, &function_type, sizeof(int), cudaMemcpyHostToDevice);
        // ##############################################


        tstart = get_time_msec();
        ProgrammableBootstrap(c2.tlwedevices[st->device_id()], c1.tlwedevices[st->device_id()], st->st(), st->device_id(),
                            encoder_device, encoder_device, function_type_d);
        tend = get_time_msec();
        ts.push_back(tend-tstart);

        CtxtCopyD2H(c2, *st);

        Synchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&et, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        double dec = TFHEpp::tlweSymDecryptDecode<lvl0param>(c2.tlwehost, sk->key.lvl0, encoder_host);
        printf("res: %3.3f, %3.3f\n", d, dec);
        st->Destroy();
        delete st;
        CleanUp();  // essential to clean and deallocate data
    }

    double tot = 0.;
    for(int i=0; i<ts.size(); i++){
        tot += ts[i];
    }
    printf("avg bs time: %f\n", tot/ts.size());
}



int main()
{
    //test_nand();
    mytest();
}
