//#define EIGEN_USE_BLAS
//#define EIGEN_USE_LAPACKE
#define NDEBUG
#define EIGEN_NO_DEBUG
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <cstdlib>
#include <float.h>
#include <random>
#include <stdint.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

#define _USE_MATH_DEFINES
using namespace std;
using namespace Eigen;

double eps = 0.3;
double mu = 12.0;
double dt = 0.001;
double beta = 0.0;

double normpdf(const Vector2d& x, const Vector2d& mu, const double sigma)
{
    Vector2d tmp = x-mu;
    return exp(-tmp.squaredNorm()/(2*sigma*sigma))/(2*M_PI*sigma*sigma);
}

Vector2d fun(const Vector2d& X)
{
    Vector2d tmp;
    tmp(0) = X(0) - pow(X(0),3)/3.0 - X(1);
    tmp(1) = X(0)/mu;
    return tmp;
}

Vector2d reflection(const Vector2d& X1, const Vector2d& X2, const Vector2d& rnd1)
{
    Vector2d e = X1 - X2;
    e /= e.norm();
    Matrix2d P = Matrix2d::Identity() - 2*e*e.transpose();
    return P*rnd1;
    
}

int main(int argc, char* argv[])
{
    struct timeval t1, t2;
    gettimeofday(&t1,NULL);
    string filename = "VDP4_0";
    if(argc > 1)
    {
         mu = 1.0*atoi(argv[1]);
         filename += argv[1];
    }
    filename += ".txt";
    int Sample_size = 2000;
    cout<<"VDP, mu = "<<mu<<endl;
    cout<<filename<<endl;
    ofstream myfile, myfile2;
    //myfile.open(filename);
    myfile2.open("loc.txt");
    int N_threads = 1;
    vector<int> syn_time(Sample_size);
    int T = 100;
    Vector2d X0;
    X0 << 0, 0;
#pragma omp parallel num_threads(N_threads)
    {
        int rank = omp_get_thread_num();
        int size = omp_get_num_threads();
        random_device rd;
        mt19937 mt(rd() + rank);
        normal_distribution<double> nm(0.0, 1.0);
        uniform_real_distribution<double> u(0, 1.0);
        Vector2d X1, X2;
        Vector2d Y0;
        Y0 << 4*u(mt)-2, 4*u(mt)-2;
    #pragma omp for
        for(int i = 0; i < Sample_size; i++)
        {
            X1 = X0;
            for(int j = 0; j < T; j++)
            {
                Vector2d rnd;
                rnd << nm(mt), nm(mt);
                Y0 += dt*fun(Y0) + eps*sqrt(dt)*rnd;
            }
            X2 = Y0;
            //cout<<"initial condition = "<<X1.transpose()<<" "<<X2.transpose()<<endl;
            int count = 0;
            int flag = 0;
            double px, qx, py, qy;
            Vector2d tmp1, tmp2, tmp3;
            while(count < 1e8 && flag == 0)
            {
                count++;
                Vector2d rnd1, rnd2;
                rnd1 << nm(mt), nm(mt);
                if(u(mt) < beta)
                {
                    rnd2 << nm(mt), nm(mt);
                }
                else
                {
                    rnd2 = reflection(X1, X2, rnd1);
                }//reflection coupling
                //rnd2 << nm(mt), nm(mt);//independent coupling
                //rnd2 = rnd1;//synchronous coupling
                X1 += dt*fun(X1);
                X2 += dt*fun(X2);
                tmp1 = eps*sqrt(dt)*rnd1;
                tmp2 = eps*sqrt(dt)*rnd2;
                tmp3 = X1 - X2;
                //cout<<tmp3.norm()<<endl;
                if(tmp3.norm() < 2*eps*sqrt(dt))
                {
                    rnd2 << nm(mt), nm(mt);
                    tmp2 = eps*sqrt(dt)*rnd2;
                    px = normpdf(X1 + tmp1, X1, eps*sqrt(dt));
                    qx = normpdf(X1 + tmp1, X2, eps*sqrt(dt));
                    py = normpdf(X2 + tmp2, X1, eps*sqrt(dt));
                    qy = normpdf(X2 + tmp2, X2, eps*sqrt(dt));
                    if(u(mt)*px < qx)
                    {
                        flag = 1;
                        X1 += tmp1;
                        X2 = X1;
                        myfile2<<X1.transpose()<<endl;
                    }
                    else
                    {
                        X1 += tmp1;
                        while(u(mt)*qy <= py)
                        {
                            rnd2 << nm(mt), nm(mt);
                            tmp2 = eps*sqrt(dt)*rnd2;
                            qy = normpdf(X2 + tmp2, X2, eps*sqrt(dt));
                            py = normpdf(X2 + tmp2, X1, eps*sqrt(dt));
                        }
                        X2 += tmp2;
                    }
                }
                else
                {
                    X1 += tmp1;
                    X2 += tmp2;
                }
                //cout<<"x1: "<<X1.transpose()<<endl<<"x2: "<<X2.transpose()<<endl;
                //myfile<<X1.transpose()<<" "<<X2.transpose()<<endl;
                
            }
            //cout<<"coupled at time"<<count<<endl;
            syn_time[i] = count;
        }
        
    }
    //cout<<syn_time[0]<<endl;
    sort(syn_time.begin(), syn_time.end());
    int MAX = syn_time[syn_time.size() - 1];
    cout<<"MAX = "<<MAX<<endl;
    vector<int> distribution(floor(MAX) + 1);
    int time_count = 0;
    int M = 20;//count the coupling time every M steps
    distribution[0] = Sample_size;
    for(int i = 0; i < (int)(MAX/M) +1; i++)
    {
        while(syn_time[time_count]/M == i)
        {
            time_count++;
        }
        distribution[i+1] = Sample_size - time_count;
//        cout<<syn_time[i]<<endl;
    }
    for(int i = 0; i <= (int)(MAX/M) +1; i++)
    {
//        cout<<distribution[i]<<endl;
        //myfile<<distribution[i]<<endl;
    }
    //myfile.close();
    myfile2.close();
    gettimeofday(&t2, NULL);
    double delta = ((t2.tv_sec  - t1.tv_sec) * 1000000u +
                    t2.tv_usec - t1.tv_usec) / 1.e6;
    
    cout << "total CPU time = " << delta <<endl;
}
