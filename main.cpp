#include <complex>
#include <fftw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>

int main()
{
    const int N = 64;
    double *x_ = new double[N];
    double *x_invers = new double[N];
    fftw_complex *in = new fftw_complex[N];
    fftw_complex *out = new fftw_complex[N];
    std::vector<double> data;
    double Xmin = -10.;
    double Xmax = 10.;
    double h = (Xmax - Xmin)/N;
    double X = Xmin;
    int i = 0;
    int count = 0;
    while(X < Xmax)
    {
        x_[i] = X;
        in[i][0] = exp(((-1.)*X*X)/2.);
        in[i][1] = 0;
        //result.push_back(exp(((-1.)*X*X)/2.));
        data.push_back((1./sqrt(2*M_PI))*exp(((-1.)*X*X)/2.));
        X += h;
        i++;
        count++;
    }
    double x_in = Xmin;
    int j = 0;
    while(x_in <= Xmax)
    {
        if(x_in < 0)
        {
            x_invers[j] = x_in + Xmax;
        }
        else
        {
            x_invers[j] = x_in + Xmin;
        }
        j++;
        x_in += h;
    }
    std::cout << count << std::endl;
    fftw_plan my_plan;
    /*my_plan = fftw_plan_r2r_1d(N, in, out, FFTW_REDFT00, FFTW_ESTIMATE);
    my_plan=fftw_plan_dft_1d(result.size(), (fftw_complex*) &result[0], \
            (fftw_complex*) &result[0], FFTW_FORWARD, FFTW_ESTIMATE);*/
    my_plan=fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(my_plan);
    std::ofstream _out("fft_gausse_test.txt");
    _out.precision(4);
    for(int i=0; i < N; ++i)
    {
       //_out<<std::fixed<<result[i]<<"\t"<<data[i]<<"\t"<<x_[i]<<std::endl;
       //_out<<std::fixed<<out[i][0]/(N/M_PI)<<"\t"<<data[i]<<"\t"<<x_[i]<<std::endl;
        _out<<std::fixed<<data[i]<<"\t"<<x_[i]<<std::endl;
    }

    std::vector<std::complex<double>> fftw_data_and_bias;
    double bias = (Xmax-Xmin)/2.;
    const std::complex<double> I(0, 1);
    for(int i = 0; i < N; i++)
    {
        std::complex<double> x(out[i][0]/sqrt(2*M_PI*N), out[0][i]/sqrt(2*M_PI*N));
        fftw_data_and_bias.push_back(x*std::exp(I*x*bias));
    }
    std::vector<std::complex<double>> coord_and_data_invers(N);
    int k = 0;
    for(int i = N/2+1; i < N; i++)
    {
        coord_and_data_invers[k] = fftw_data_and_bias[i];
        k++;
    }
    for(int i = 0; i <= N/2; i++)
    {
        coord_and_data_invers[k] = fftw_data_and_bias[i];
        k++;
    }
    std::ofstream _out_("fft_gausse_our.txt");
    _out_.precision(4);
    for(int i = 0; i < N; i++)
    {
        _out_<<std::fixed<<coord_and_data_invers[i].real()<<'\t'<<x_[i]<<
               '\t'<<coord_and_data_invers[i].imag()<<std::endl;
    }

    fftw_destroy_plan(my_plan);

    delete [] out;
    delete [] in;
    delete [] x_invers;
    delete [] x_;

    return 0;
}