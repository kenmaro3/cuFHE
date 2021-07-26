#pragma once

#include <stdint.h>
#include <stdio.h>
#include <cmath>
#include <iostream>

#include "cufhe.h"

using namespace std;

namespace cufhe {


class EncoderDevice
{
    public:
        double a;  // input lower bound
        double b;  // input upper bound
        double effective_b;
        double d;
        double d2;
        double half;
        double half_d;
        int bp;    // bit precision including noise bit (lvl0param::T - bp is padding bit)
        // bp = (noise bit + plaintext precision bit)
        bool is_type_second;
        double mult_number;


        __device__
        void print(){
            printf("=========================================\n");
            printf("\nEncoder Print\n");
            printf("a     : %f\n", this->a);
            printf("b     : %f\n", this->b);
            printf("effective_b     : %f\n", this->effective_b);
            printf("d     : %f\n", this->d);
            printf("half  : %f\n", this->half);
            printf("half_d: %f\n", this->half_d);
            printf("bp    : %d\n", this->bp);
            printf("type    : %d\n", this->is_type_second);

        }

        //__device__
        //static EncoderDevice copy(EncoderDevice &encoder){
        //    
        //    if(encoder.is_type_second){
        //        Encoder tmp(encoder.a, abs(encoder.a), encoder.bp, encoder.is_type_second);
        //        return tmp;
        //    }else{
        //        Encoder tmp(encoder.a, encoder.b, encoder.bp, encoder.is_type_second);
        //        return tmp;
        //    }
        //}

        __device__
        EncoderDevice(){
        };

        __host__ __device__
        EncoderDevice(double a, double b, int bp, bool is_type_second=true){

            if(is_type_second){
                this->a = a;
                this->effective_b = b;
                double tmp = b-a;
                this->b = b + tmp;
                this->d = this->b-this->a;
                this->half_d = (this->b-this->a)/2.;
                this->half = (this->b+this->a)/2.;
                this->bp = bp;
                this->is_type_second = true;
            }else{
                this->a = a;
                this->b = b;
                this->d = b-a;
                this->d2 = b-a;
                this->half_d = (b-a)/2.;
                this->half = (b+a)/2.;
                this->bp = bp;
                this->is_type_second = false;
            }


        }


        __host__ __device__
        void update(double a, double b, int bp){
            //this->a = a;
            //this->b = b;
            //this->d = b-a;
            //this->d2 = b-a;
            //this->half_d = (b-a)/2.;
            //this->half = (b+a)/2.;
            //this->bp = bp;

            if(this->is_type_second){
                this->a = a;
                this->b = b;
                this->effective_b = abs(this->a);
                this->d = this->b-this->a;
                this->half_d = (this->b-this->a)/2.;
                this->half = (this->b+this->a)/2.;

                this->bp = bp;
            }
        }


        __host__ __device__
        void update(double x){
            if(this->is_type_second){
                this->a = this->a*x;
                this->b = this->b*x;
                this->effective_b = abs(this->a);
                this->d = this->b-this->a;
                this->half_d = (this->b-this->a)/2.;
                this->half = (this->b+this->a)/2.;
            }
        }

        __host__ __device__
        double encode_0_1(double x) const{
            return (x-this->a)/this->d;

        }

        __host__ __device__
        static lvl0param::T dtotx(double d, int bpx){
            double tmp = d - floor(d);
            tmp = tmp * pow(2., bpx);
            double tmp2 = tmp - floor(tmp);
            if(tmp2 < 0.5){
                return static_cast<lvl0param::T>(tmp);
            }else{
                return static_cast<lvl0param::T>(tmp+1);
            }

        }

        __host__ __device__
        static lvl0param::T dtotx(double d, double max,  int bpx){
            d = d/max;
            double tmp = d - floor(d);
            tmp = tmp * pow(2., bpx);
            double tmp2 = tmp - floor(tmp);
            if(tmp2 < 0.5){
                return static_cast<lvl0param::T>(tmp);
            }else{
                return static_cast<lvl0param::T>(tmp+1);
            }
        }

        __host__ __device__
        lvl0param::T dtotx(double d) const{
            //return static_cast<lvl0param::T>(int64_t((d - int64_t(d)) * (1LL << this->bp)));
            double tmp = d - floor(d);
            tmp = tmp * pow(2., this->bp);
            double tmp2 = tmp - floor(tmp);
            if(tmp2 < 0.5){
                return static_cast<lvl0param::T>(tmp);
            }else{
                return static_cast<lvl0param::T>(tmp+1);
            }
        }

        __device__
        lvl0param::T encode(double x) const{
            //printf("here: %f\n", x);
            assert(x >= this->a);
            assert(x <= this->b);
            return dtotx((x-this->a)/this->d);
        }

        __host__ __device__
        double txtod(lvl0param::T x) const{
            double tmp_0_1 = static_cast<double>(x) / pow(2, this->bp);
            return tmp_0_1;
        }


        __host__ __device__
        double t32tod(lvl0param::T x) const{
            double tmp_0_1 = static_cast<double>(x) / pow(2, std::numeric_limits<lvl0param::T>::digits);
            return tmp_0_1;
        }

        __host__ __device__
        double decode(const lvl0param::T x){
            double tmp_0_1 = this->txtod(x);
            //printf("tmp_0_1: %f\n", tmp_0_1);
            tmp_0_1 = tmp_0_1 - floor(tmp_0_1);
            return tmp_0_1 * this->d + this->a;
        }


        __device__
        double identity_function(double x){
            return x;
        }

        __device__
        double relu_function(double x){
            return x >= 0 ? x : 0.; 
        }

        __device__
        double sigmoid_function(double x){
            return 1./(1.+pow(std::exp(1.0), x*(-1.))); 
        }
        
        __device__
        double mult_function(double x){
            assert(this->mult_number != NULL);
            return x * this->mult_number; 
        }

}; // class Encoder

} // namespace cufhe
