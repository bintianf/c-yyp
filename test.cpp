// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE
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

double eps = 1;
double dt = 0.001;
int T = 1000;
double corner = 0;
double range = 1;

double normpdf(double x, double mu, double sigma)
{
    double tmp; 
    tmp = x - mu;
    return exp(-(tmp*tmp)/(2*sigma*sigma))/(sqrt(2*M_PI)*sigma);
}

double fun(double x, double alpha_1, double a, double alpha_2, double beta)
{
    double tmp = alpha_1*a*x - alpha_2*x*x - beta*x;
    return tmp;
}


int main()
{
    int Sample_size = 1000;
    vector<int> coupling_time(Sample_size);
    // ofstream myfile;
    // myfile.open("keizer.txt");
    // int N_threads = 8;
// #pragma omp parallel num_threads(N_threads)
    // {
        // int rank = omp_get_thread_num();
        // int size = omp_get_num_threads();
        random_device rd;
        mt19937 mt(rd());
        normal_distribution<double> nm(0.0, 1.0);
        uniform_real_distribution<double> u(0, 1.0);
    for(int i = 0; i < Sample_size; i++)
        {
            double x1, x2 ;
            x1 = corner + range*u(mt);
            x2 = corner + range*u(mt);
            int count = 0;
            int flag = 0;
            while(count < 10000 && flag == 0)
            {
                count++;
                double rnd1, rnd2, temp1, temp2, temp3;
                double min1, max1, min2, max2, swap;
                rnd1 = nm(mt);
                rnd2 = -rnd1;//reflection coupling
                temp1 = eps*sqrt(dt)*rnd1;
                temp2 = eps*sqrt(dt)*rnd2;
                temp3 = abs(x1 - x2);
                x1 += dt*fun(x1, 0.05, 0.2, 0.05, 0.05); // parameters of reaction rates
                x2 += dt*fun(x2, 0.05, 0.2, 0.05, 0.05); 
                cout <<"i=    " << i <<  "x1   " << x1 << "x2    "  << x2 << endl;
                if(temp3 < 2*eps*sqrt(dt))
                {
                    min1 = normpdf(x1 + temp1, x2, eps*sqrt(dt));
                    min2 = normpdf(x2 + temp2, x1, eps*sqrt(dt));
                    max1 = normpdf(x1, x1 + temp1, eps*sqrt(dt));
                    max2 = normpdf(x2, x2 + temp2, eps*sqrt(dt));
                    cout << "*************"<< endl;
                    cout << "min1=" << min1 << "     min2=" << min2 <<endl;
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
                        x1 += temp1;
                        x2 = x1;
                    }
                    else
                    {
                        x1 += temp1;
                        x2 += temp2;
                    }
                }
                else
                {
                    x1 += temp1;
                    x2 += temp2;
                }
            cout << "flag=====" << flag << endl;
            cout << "111111111     " << x1 << "  22222222222222222     " << x2 << endl;
            }
            coupling_time[i] = count;
            cout << " coupling time  " <<  coupling_time[i] << endl;
        }
        
    // }

    int MAX = coupling_time[coupling_time.size() - 1];
    cout<<"MAX = "<<MAX<<endl;
    vector<int> distribution(floor(MAX) + 1);
    int time_count = 0;
    int M = 5;//count the coupling time every M steps
    distribution[0] = Sample_size;
    for(int i = 0; i < (int)(MAX/M) +1; i++)
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
       cout<<distribution[i]<<endl;
        // myfile<< i*M*dt << " , " << distribution[i]/Sample_size<<endl;
    }
     
    // myfile.close();
}