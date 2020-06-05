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
#include <Eigen/Dense>

#define _USE_MATH_DEFINES
using namespace std;
using namespace Eigen;

double eps = 0.05;
double dt = 0.001;
int T = 1000;
double corner = 0;
double range = 1;
double a = 2;
double b = 0.4;

double normpdf(double x, double mu, double sigma)
{
    double tmp = x - mu;
    return  exp(-tmp*tmp/(2*sigma*sigma ))/sqrt(2*M_PI*sigma*sigma);
}


double fun(double x)
{
    double tmp = a - b*x*x;
    return tmp;
}

double sigma(double x)
{
    double tmp = a + b*x*x;
    return sqrt(tmp);
}
int main()
{
    int Sample_size = 100;
    vector<int> coupling_time(Sample_size);
    ofstream myfile;
    myfile.open("newfunction.txt");
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
            double x1, x2;
            x1 = corner + range*u(mt);
            x2 = corner + range*u(mt);
            int count = 0;
            int flag = 0;
            while(count < 1e8 && flag == 0)
            {
                count++;
                double rnd1, rnd2, temp1, temp2, temp3, temp4;
                double px, qx, py, qy, swap;
                rnd1 = nm(mt);
                rnd2 = -rnd1;//reflection coupling
                x1 += dt*fun(x1); 
                x2 += dt*fun(x2);
                temp1 = sigma(x1)*sqrt(dt)*rnd1;
                temp2 = sigma(x2)*sqrt(dt)*rnd2;
                temp3 = x1 - x2;
                temp4 = min(sigma(x1), sigma(x2));
                if(std::abs(temp3) < 2*temp4*sqrt(dt))
                {
                    rnd2 = nm(mt);
                    temp2 = sigma(x2)*sqrt(dt)*rnd2;
                    px = normpdf(x1 + temp1, x1, sigma(x1)*sqrt(dt));
                    qx = normpdf(x1 + temp1, x2, sigma(x2)*sqrt(dt));
                    py = normpdf(x2 + temp2, x1, sigma(x1)*sqrt(dt));
                    qy = normpdf(x2 + temp2, x2, sigma(x2)*sqrt(dt));
                    if(u(mt)*px < qx)
                    {
                        flag = 1;
                        x1 += temp1;
                        x2 = x1;
                    }
                    else
                    {
                        x1 += temp1;
                        while(u(mt)*qy <= py)
                        {
                            rnd2 = nm(mt);
                            temp2 = sigma(x2)*sqrt(dt)*rnd2;
                            qy = normpdf(x2 + temp2, x2, sigma(x2)*sqrt(dt));
                            py = normpdf(x2 + temp2, x1, sigma(x1)*sqrt(dt));
                        }
                        x2 += temp2;
                    }                    
                
                }
                else
                {
                    x1 += temp1;
                    x2 += temp2;
                }
            }

            coupling_time[i] = count;
            //cout << "i =    " << i <<"    time = "<<coupling_time[i] << endl;
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
