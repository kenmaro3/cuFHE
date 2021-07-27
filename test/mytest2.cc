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
#include <cassert>
using namespace std;

#include <chrono>
using namespace std::chrono;
inline double get_time_msec(void){
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count())/1000000;
}

void HomADDFixedEncoder(TLWE<lvl0param> &res, const TLWE<lvl0param> &ca, const TLWE<lvl0param> &cb, Encoder &encoder1, Encoder &encoder2)
{
    assert(encoder1.a == encoder2.a);
    assert(encoder1.b == encoder2.b);
    assert(encoder1.bp == encoder2.bp);
    for (int i = 0; i <= lvl0param::n; i++) res[i] = ca[i] + cb[i];
    //res[lvl0param::n] += encoder1.dtotx(0.5);
    res[lvl0param::n] -= encoder1.encode(0.);
}

void HomADDCONST(TLWE<lvl0param> &res, const TLWE<lvl0param> &ca, const double &b, Encoder &encoder)
{
    for (int i = 0; i < lvl0param::n; i++) res[i] = ca[i];
    if(b>0){
        lvl0param::T tmp = encoder.encode(b + encoder.a);
        res[lvl0param::n] = ca[lvl0param::n] + tmp;
    }else{
        lvl0param::T tmp = encoder.encode(-b + encoder.a);
        res[lvl0param::n] = ca[lvl0param::n] - tmp;

    }
}

void print_vec_1d(vector<double> x){
    for(int i=0; i<x.size(); i++){
        printf("%f, ", x[i]);
    }
    printf("\n");
}

void print_vec_2d(vector<vector<double>> x){
    for(int i=0; i<x.size(); i++){
        print_vec_1d(x[i]);
    }
    printf("\n");
}

vector<double> give_me_vector(int size){
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::normal_distribution<> dist(0.0, 1.0);
  vector<double> res;
  for (std::size_t n = 0; n < size; ++n) {
    double tmp = dist(engine);
    res.push_back(tmp);
  }
  return res;
}

vector<vector<double>> give_me_weight(int dout, int din){
  vector<vector<double>> res;
  for (std::size_t n = 0; n < dout; ++n) {
    res.push_back(give_me_vector(din));
  }
  return res;
}

template <class P> 
vector<Ctxt> encrypt_vector(const vector<double> xs, const array<typename P::T, P::n> sk, Encoder &encoder){
    vector<Ctxt> res(xs.size());
    for(int i=0; i<xs.size(); i++){
        res[i].tlwehost = TFHEpp::tlweSymEncodeEncrypt<lvl0param>(xs[i], lvl0param::alpha, sk, encoder);
    }
    return res;
}

template <class P>
vector<double> decrypt_vector(const vector<TLWE<P>> cs, const array<typename P::T, P::n> sk, Encoder &encoder, bool is_print=true){
  vector<double> res;
  for(int i=0; i<cs.size(); i++){
    res.push_back(TFHEpp::tlweSymDecryptDecode<P>(cs[i], sk, encoder));
  }

  if(is_print){
    print_vec_1d(res);
  }
  return res;
}

double get_max(vector<double> x){
  double res = abs(x[0]);
  for(int i=1; i<x.size(); i++){
    if(abs(x[i]) > res){
      res = abs(x[i]);
    }
  }
  return res;
}

double get_wider(double x){
  if(x>=0){
    //return ceil(x);
    return x+0.2;
  }else{
    //double tmp = ceil(abs(x));
    double tmp = abs(x)+0.2;
    return tmp*(-1.0);
  }
}

template <class P>
vector<double> decrypt_vector(const vector<Ctxt>& cs, const array<typename P::T, P::n> sk, Encoder &encoder, bool is_print=true){
  vector<double> res;
  for(int i=0; i<cs.size(); i++){
    res.push_back(TFHEpp::tlweSymDecryptDecode<P>(cs[i].tlwehost, sk, encoder));
  }

  if(is_print){
    print_vec_1d(res);
  }
  return res;
}

double get_max_ds(vector<double> ds, double expansion){
    double max_ds = get_max(ds);
    max_ds = get_wider(max_ds);
    max_ds *= expansion;
    return max_ds;
}

double get_max_ds(vector<vector<double>> ds, double expansion){
    double max_ds = 0.;
    for(int i=0; i<ds.size(); i++){
        max_ds = max(max_ds, get_max(ds[i]));
    }
    max_ds = get_wider(max_ds);
    max_ds *= expansion;
    return max_ds;
}


vector<Ctxt> vector_mult(vector<Ctxt>& cs, const vector<double> ds, Stream* st, EncoderDevice *encoder_domain, EncoderDevice *encoder_target, Encoder &encoder_host, double expansion=1.0, bool is_adjust_encoder=true)
{
    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    vector<Ctxt> res(cs.size());
    cufhe::EncoderDevice *encoder_domain_d, *encoder_target_d;
    //EncoderDevice encoder_domain = EncoderDevice::copy(encoder);
    //EncoderDevice encoder_target = EncoderDevice::copy(encoder);

    if(is_adjust_encoder){
        double max_ds = get_max_ds(ds, expansion);
        encoder_target->update(max_ds);
        encoder_host.update(max_ds);
        //encoder->update(max_ds);
    }

    int function_type = 3;
    int * function_type_d;
    cudaMalloc((void **)&function_type_d, sizeof(int));
    cudaMemcpy(function_type_d, &function_type, sizeof(int), cudaMemcpyHostToDevice);

    assert(cs.size()==ds.size());
    for(int i=0; i<cs.size(); i++){
        //printf("loop %d\n", i);
        cudaSetDevice(st->device_id());
        CtxtCopyH2D(cs[i], *st);
        // ##############################################
        encoder_target->mult_number = ds[i];
        cudaMalloc((void **)&encoder_domain_d, sizeof(EncoderDevice));
        cudaMemcpy(encoder_domain_d, encoder_domain, sizeof(EncoderDevice), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&encoder_target_d, sizeof(EncoderDevice));
        cudaMemcpy(encoder_target_d, encoder_target, sizeof(EncoderDevice), cudaMemcpyHostToDevice);
        // ##############################################
        ProgrammableBootstrap(res[i].tlwedevices[st->device_id()], cs[i].tlwedevices[st->device_id()], st->st(), st->device_id(),
                            encoder_domain_d, encoder_target_d, function_type_d);

        CtxtCopyD2H(res[i], *st);
        // ##############################################
    }
    Synchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return res;
}


TFHEpp::TLWE<lvl0param> vector_sum_in_col(vector<Ctxt>& cs, Encoder &encoder, double expansion=1.0){
    encoder.update(expansion);
    Ctxt res;
    res.tlwehost = cs[0].tlwehost;
    for(int i=1; i<cs.size(); i++){
        HomADDFixedEncoder(res.tlwehost, res.tlwehost, cs[i].tlwehost, encoder, encoder);
    }
    return res.tlwehost;
}


TFHEpp::TLWE<lvl0param> inner(vector<Ctxt>& cs, const vector<double> ds, Stream* st, EncoderDevice *encoder_domain, EncoderDevice *encoder_target, Encoder &encoder_host, double expansion=1.0, bool is_adjust_encoder=true)
{
    if(is_adjust_encoder){
        double max_ds = get_max_ds(ds, expansion);
        encoder_target->update(max_ds);
        encoder_host.update(max_ds);
    }
    vector<Ctxt> tmp1 = vector_mult(cs, ds, st, encoder_domain, encoder_target, encoder_host, 1., false);
    TLWE<lvl0param> tmp2 = vector_sum_in_col(tmp1, encoder_host, 1.);
    return tmp2;
}

void wtx(vector<Ctxt>& res, vector<Ctxt>& cs, const vector<vector<double>> w, Stream* st, EncoderDevice *encoder, Encoder &encoder_host, double expansion=1.0, bool is_adjust_encoder=true)
{
    cufhe::EncoderDevice *encoder_domain_d, *encoder_target_d;
    EncoderDevice encoder_domain = EncoderDevice::copy(encoder);
    EncoderDevice encoder_target = EncoderDevice::copy(encoder);

    double max_ds;
    if(is_adjust_encoder){
        max_ds = get_max_ds(w, expansion);
        encoder_target.update(max_ds);
        encoder_host.update(max_ds);
    }
    //vector<Ctxt> res(cs.size());
    //vector<TFHEpp::TLWE<P>> res;
    for(int i=0; i<w.size(); i++){
        res[i].tlwehost = inner(cs, w[i], st, &encoder_domain, &encoder_target, encoder_host, 1., false);
        //res.push_back(inner(cs, w[i], st, &encoder_domain, &encoder_target, encoder_host, 1., false));
    }
    encoder->update(max_ds);
}

void btx(vector<Ctxt>& res, vector<Ctxt>& cs, const vector<double> b, Encoder &encoder_host)
{
    assert(res.size()==cs.size());
    for(int i=0; i<cs.size(); i++){
        HomADDCONST(res[i].tlwehost, cs[i].tlwehost, b[i], encoder_host);
        //HomADDCONST(res[i], cs[i], b[i], encoder_host);
    }
}

double inner(const vector<double> x, const vector<double> y, bool is_print=true){
    assert(x.size() == y.size());
    double res = 0;
    for(std::size_t i=0; i<x.size(); i++){
        res += x[i]*y[i];
    }
    if(is_print){
        printf("%f\n", res);
    }
    return res;
}

vector<double> wtx(const vector<double> x, const vector<vector<double>> w, bool is_print=true){
    assert (x.size() == w[0].size());
    vector<double> res;
    for(int i=0; i<w.size(); i++){
        res.push_back(inner(x, w[i], false));
    }
    if(is_print){
        print_vec_1d(res);
    }

    return res;
}

vector<double> btx(const vector<double> x, const vector<double> b, bool is_print=true){
    assert(x.size()==b.size());
    vector<double> res;
    for(int i=0; i<x.size(); i++){
        res.push_back(x[i]+b[i]);
    }
    if(is_print){
        print_vec_1d(res);
    }
    return res;
}

vector<double> relu(const vector<double> x, bool is_print=true){
    vector<double> res;
    for(int i=0; i<x.size(); i++){
        if(x[i]>=0){
        res.push_back(x[i]);
        }else{
        res.push_back(0);
        }
    }
    if(is_print){
        print_vec_1d(res);
    }
    return res;
}

void relu(vector<Ctxt>& res, vector<Ctxt>& cs, Stream* st, EncoderDevice *encoder, Encoder &encoder_host)
{
    assert(res.size()==cs.size());
    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cufhe::EncoderDevice *encoder_domain_d, *encoder_target_d;
    EncoderDevice encoder_domain = EncoderDevice::copy(encoder);
    EncoderDevice encoder_target = EncoderDevice::copy(encoder);

    int function_type = 1;
    int * function_type_d;
    cudaMalloc((void **)&function_type_d, sizeof(int));
    cudaMemcpy(function_type_d, &function_type, sizeof(int), cudaMemcpyHostToDevice);

    for(int i=0; i<cs.size(); i++){
        cudaSetDevice(st->device_id());
        CtxtCopyH2D(cs[i], *st);
        // ##############################################
        cudaMalloc((void **)&encoder_domain_d, sizeof(EncoderDevice));
        cudaMemcpy(encoder_domain_d, &encoder_domain, sizeof(EncoderDevice), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&encoder_target_d, sizeof(EncoderDevice));
        cudaMemcpy(encoder_target_d, &encoder_target, sizeof(EncoderDevice), cudaMemcpyHostToDevice);
        // ##############################################
        ProgrammableBootstrap(res[i].tlwedevices[st->device_id()], cs[i].tlwedevices[st->device_id()], st->st(), st->device_id(),
                            encoder_domain_d, encoder_target_d, function_type_d);

        CtxtCopyD2H(res[i], *st);
        // ##############################################
    }
    Synchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

double avg(vector<double> x){
    double res = 0;
    for(int i=0; i<x.size(); i++){
        res += x[i];
    }
    return res/double(x.size());
}

int main(){
    printf("hello, world\n");

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const uint32_t kNumSMs = prop.multiProcessorCount;
    const uint32_t kNumTests = kNumSMs * 32;   // * 8;
    constexpr uint32_t kNumLevels = 10;  // Gate Types, Mux is counted as 2.

    // ###########################################
    // model side parameters
    printf("\nprepare_model=============================================================\n");
    int d1 = 5;
    int d2 = 3;
    int d3 = 1;
    string nl_type1 = "sigmoid";
    string nl_type2 = "sigmoid";
    vector<double> d;

    vector<double> x1 = give_me_vector(d1);
    vector<vector<double>> w = give_me_weight(d2, d1);
    vector<double> b = give_me_vector(d2);
    vector<vector<double>> w2 = give_me_weight(d3, d2);
    vector<double> b2 = give_me_vector(d3);

    printf("\n=============================================================\n");
    printf("x1\n");
    print_vec_1d(x1);
    printf("w\n");
    print_vec_2d(w);
    printf("b\n");
    print_vec_1d(b);

    printf("\n=============================================================\n");
    printf("w\n");
    print_vec_2d(w2);
    printf("b\n");
    print_vec_1d(b2);


    // ###########################################
    // encoder side parameters
    printf("\n=============================================================\n");
    double encoder_a = -3.;
    double encoder_b = 3.;
    int bs_bp = 32;

    cufhe::EncoderDevice *encoder_domain_d, *encoder_target_d;
    TFHEpp::Encoder encoder_host(encoder_a, encoder_b, bs_bp);
    cufhe::EncoderDevice* encoder = new EncoderDevice(encoder_a, encoder_b, bs_bp);

    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk);
    TFHEpp::GateKey* gk2 = new TFHEpp::GateKey(*sk);

    cout << "n:" << sk->params.lvl0.n << endl;


    vector<Ctxt> c1 = encrypt_vector<lvl0param>(x1, sk->key.lvl0, encoder_host);
    printf("just decrypt\n");
    decrypt_vector<lvl0param>(c1, sk->key.lvl0, encoder_host);


    //###########################################################
    Synchronize();

    Initialize(*gk);  // essential for GPU computing

    Stream* st = new Stream;
    st->Create();

    double tstart, tend;
    vector<double> ts;
    bool is_decrypt = false;
    for(int i=0; i<3; i++){
        tstart = get_time_msec();

        printf("\n111111111=============================================================\n");
        printf("cipher=============================================================\n");
        vector<Ctxt> c2(d2);
        wtx(c2, c1, w, st, encoder, encoder_host, 1., true);
        if(is_decrypt)
            printf("wtx: ");
            decrypt_vector<lvl0param>(c2, sk->key.lvl0, encoder_host);
        btx(c2, c2, b, encoder_host);
        if(is_decrypt)
            printf("btx: ");
            decrypt_vector<lvl0param>(c2, sk->key.lvl0, encoder_host);
        relu(c2, c2, st, encoder, encoder_host);
        if(is_decrypt)
            printf("relu: ");
            decrypt_vector<lvl0param>(c2, sk->key.lvl0, encoder_host);

        printf("\nraw debug=============================================================\n");
        printf("wtx: ");
        vector<double> tmp = wtx(x1, w);
        printf("btx: ");
        tmp = btx(tmp, b);
        printf("relu: ");
        tmp = relu(tmp);

        printf("\n222222222=============================================================\n");
        printf("cipher=============================================================\n");
        vector<Ctxt> c3(d3);
        wtx(c3, c2, w2, st, encoder, encoder_host, 1., true);
        if(is_decrypt)
            printf("wtx: ");
            decrypt_vector<lvl0param>(c3, sk->key.lvl0, encoder_host);
        btx(c3, c3, b2, encoder_host);
        if(is_decrypt)
            printf("btx: ");
            decrypt_vector<lvl0param>(c3, sk->key.lvl0, encoder_host);
        relu(c3, c3, st, encoder, encoder_host);
        if(is_decrypt)
            printf("relu: ");
            decrypt_vector<lvl0param>(c3, sk->key.lvl0, encoder_host);

        printf("\nraw debug=============================================================\n");
        printf("wtx: ");
        tmp = wtx(tmp, w2);
        printf("btx: ");
        tmp = btx(tmp, b2);
        printf("relu: ");
        tmp = relu(tmp);

        tend = get_time_msec();
        ts.push_back(tend-tstart);
    }
    printf("\nend=============================================================\n");
    st->Destroy();
    delete st;
    CleanUp();  // essential to clean and deallocate data
    double ts_avg = avg(ts);
    printf("time avg: %f\n", ts_avg);
    return 0;
}

