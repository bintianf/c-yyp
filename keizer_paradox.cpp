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

double eps = 0.5;
double dt = 0.001;
int T = 10000;
double corner = 0;
double range = 1;


double fun(double x, double alpha_1, double a, double alpha_2, double beta)
{
    double tmp = alpha_1*a*x - alpha_2*x*x - beta*x;
    return tmp;
}


int main()
{
    int Sample_size = 10000;
    vector<int> coupling_time(Sample_size);
    ofstream myfile;
    myfile.open("keizer.txt");
    int N_threads = 8;
#pragma omp parallel num_threads(N_threads)
    {
        int rank = omp_get_thread_num();
        int size = omp_get_num_threads();
        random_device rd;
        mt19937 mt(rd());
        normal_distribution<double> nm(0.0, 1.0);
        uniform_real_distribution<double> u(0, 1.0);
    for(int i = rank*Sample_size/size; i < (rank+1)*Sample_size/size; i++)
        {
            double x1, x2;
            x1 = corner + range*u(mt);
            x2 = corner + range*u(mt);
            int count = 0;
            int flag = 0;
            while(count < 1e6 && flag == 0)
            {
                count++;
                double rnd1, rnd2, temp1, temp2, temp3;
                double min1, max1, min2, max2, swap;
                rnd1 = nm(mt);
                rnd2 = -rnd1;//reflection coupling
                temp1 = eps*sqrt(dt)*rnd1;
                temp2 = eps*sqrt(dt)*rnd2;
                temp3 = x1 - x2;
                x1 += dt*fun(x1, 0.05, 0.2, 0.05, 0.05); // parameters of reaction rates
                x2 += dt*fun(x2, -0.05, 0.2, 0.05, 0.05); 
                if(temp3 < 2*eps*sqrt(dt))
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
            coupling_time[i] = count;
            cout << "i =    " << i <<"    time = "<<coupling_time[i] << endl;
        }
    }
    sort(coupling_time.begin(), coupling_time.end());
    
    // for(int i = 0; i < Sample_size; i++)
    // {
    //     cout << "******i =    "<< coupling_time[i] << endl;
    // }
    int MAX = coupling_time[coupling_time.size() - 1];
    cout<<"MAX = "<<MAX<<endl;
    vector<double> distribution(floor(MAX) + 1);
    int time_count = 0;
    int M = 20;//count the coupling time every M steps
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
    //    cout << " i = "  << i << "*******" <<distribution[i]<<endl;
        myfile << i * M * dt <<","<< distribution[i]/Sample_size <<endl;
    }
     
    myfile.close();
}