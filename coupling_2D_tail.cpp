#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
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
#include <eigen/Eigen/Dense>

#define _USE_MATH_DEFINES
using namespace std;
using namespace Eigen;

double eps = 0.5;
double dt = 0.001;
int T = 10000;
double corner = -2;
double range = 4;

double normpdf(const Vector2d& x, const Vector2d& mu, const double sigma)
{
    Vector2d tmp = x-mu;
    return exp(-tmp.squaredNorm()/(2*sigma*sigma))/(2*M_PI*sigma*sigma);
}

Vector2d fun(const Vector2d& X)
{
    Vector2d tmp;
    double xx = X(0);
    double yy = X(1);
    double sqr = X.norm();
    tmp(0) = -4*xx*(sqr - 1) + yy;
    tmp(1) = -4*yy*(sqr - 1) - xx;
    return tmp;
}

Vector2d reflection(const Vector2d& X1, const Vector2d& X2, const Vector2d& rnd1)
{
    Vector2d e = X1 - X2;
    e /= e.norm();
    Matrix2d P = Matrix2d::Identity() - 2*e*e.transpose();
    return P*rnd1;
    
}

int main()
{
    struct timeval t1, t2;
    gettimeofday(&t1,NULL);
    int Sample_size = 1000000;
    vector<int> coupling_time(Sample_size);
    ofstream myfile;
    myfile.open("ring_tail.txt");
    int N_threads = 8;
#pragma omp parallel num_threads(N_threads)
    {
        int rank = omp_get_thread_num();
        int size = omp_get_num_threads();
        random_device rd;
        mt19937 mt(rd() + rank);
        normal_distribution<double> nm(0.0, 1.0);
        uniform_real_distribution<double> u(0, 1.0);
        for(int i = rank*Sample_size/size; i < (rank+1)*Sample_size/size; i++)
        {
            Vector2d X1, X2;
            X1 << corner + range*u(mt), corner + range*u(mt);
            X2 << corner + range*u(mt), corner + range*u(mt);
            int count = 0;
            int flag = 0;
            double max1, min1, max2, min2, swap;
            Vector2d tmp1, tmp2, tmp3;
            while(count < 1e8 && flag == 0)
            {
                count++;
                Vector2d rnd1, rnd2;
                rnd1 << nm(mt), nm(mt);
                rnd2 = reflection(X1, X2, rnd1);//reflection coupling
                //rnd2 << nm(mt), nm(mt);//independent coupling
                //rnd2 = rnd1;//synchronous coupling
                tmp1 = eps*sqrt(dt)*rnd1;
                tmp2 = eps*sqrt(dt)*rnd2;
                tmp3 = X1 - X2;
                X1 += dt*fun(X1);
                X2 += dt*fun(X2);
                //cout<<tmp3.norm()<<endl;
                if(tmp3.norm() < 2*eps*sqrt(dt))
                {
                    min1 = normpdf(X1 + tmp1, X2, eps*sqrt(dt));
                    min2 = normpdf(X2 + tmp2, X1, eps*sqrt(dt));
                    max1 = normpdf(X1, X1 + tmp1, eps*sqrt(dt));
                    max2 = normpdf(X2, X2 + tmp2, eps*sqrt(dt));
                    if(max1 < min1)
                    {
                        swap = max1;
                        max1 = min1;
                        min1 = swap;
                    }
                    if(max2 < min2)
                    {
                        swap = max2;
                        max2 = min2;
                        min2 = swap;
                    }
                    if(u(mt) < min1*min2/(max1*max2))
                    {
                        flag = 1;
                        X1 += tmp1;
                        X2 = X1;
                        //cout<<X1 + tmp1<<" "<<X2 + tmp2<<" "<<min1/max1<<" "<<min2/max2<<endl;
                    }
                    else
                    {
                        X1 += tmp1;
                        X2 += tmp2;
                    }
                }
                else
                {
                    X1 += tmp1;
                    X2 += tmp2;
                }
                //cout<<X1<<" "<<X2<<endl;
                
            }
            coupling_time[i] = count;
        }
        
    }
    
    //cout<<syn_time[0]<<endl;
    
    sort(coupling_time.begin(), coupling_time.end());
    int MAX = coupling_time[coupling_time.size() - 1];
    cout<<"MAX = "<<MAX<<endl;
    vector<int> distribution(floor(MAX) + 1);
    int time_count = 0;
    int M = 20;//count the coupling time every M steps
    distribution[0] = Sample_size;
    for(int i = 0; i < (int)(MAX/20) +1; i++)
    {
        while(coupling_time[time_count]/M == i)
        {
            time_count++;
        }
        distribution[i+1] = Sample_size - time_count;
//        cout<<syn_time[i]<<endl;
    }
    for(int i = 0; i <= MAX; i++)
    {
//        cout<<distribution[i]<<endl;
        myfile<<distribution[i]<<endl;
    }
     
    myfile.close();
    gettimeofday(&t2, NULL);
    double delta = ((t2.tv_sec  - t1.tv_sec) * 1000000u +
                    t2.tv_usec - t1.tv_usec) / 1.e6;
    
    cout << "total CPU time = " << delta <<endl;
}
