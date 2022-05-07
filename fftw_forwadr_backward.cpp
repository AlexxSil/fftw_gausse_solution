#include <iostream>
#include <fstream>
using namespace std;
#include <fftw3.h>
#include <math.h>


double Sqr(double x)
{
	return x*x;
}

double Func_begin(const double x)
{
	return exp(-Sqr(x)/2.0);
}


int main()
{
    fstream file;
    int i_t, kx;
    double x_coordinate;
    const int nx=1024;
    const double xmin=-10.0;
    const double xmax=10.0;
    const double dx=(xmax-xmin)/(double)nx;
    fftw_complex *func_c, *func_m;
    fftw_plan plan_fwd, plan_bwd;
    func_c=(fftw_complex*) fftw_malloc(nx*sizeof(fftw_complex));
    func_m=(fftw_complex*) fftw_malloc(nx*sizeof(fftw_complex));
    plan_fwd = fftw_plan_dft_1d(nx, func_c, func_m, FFTW_FORWARD, FFTW_MEASURE);
    plan_bwd = fftw_plan_dft_1d(nx, func_m, func_c, FFTW_BACKWARD, FFTW_MEASURE);
    for(kx=0; kx<nx; kx++)
    {
		x_coordinate=xmin+dx*(double)kx;
		func_c[kx][0]=Func_begin(x_coordinate);
		func_c[kx][1]=0.0;
	}
	file.open("func_coord_initial.dat", ios::out | ios::trunc);
    for(kx=0; kx<nx; kx++)
    {
		x_coordinate=xmin+dx*(double)kx;
		file << x_coordinate << " "<< func_c[kx][0] << " "<< func_c[kx][1] << endl;
	}
	file.close();
    fftw_execute(plan_fwd); // tranfsormation in momentum representation
    const double dpx=2.0*M_PI/(dx*(double)nx); // step over px in the momentum space
    double *px = new double [nx];
    for (kx=0; kx<nx/2; kx++) // filling array of momentum values for considered grid
        px[kx]=dpx*(double)kx;
    for (kx=nx/2; kx<nx; kx++)
        px[kx]=-dpx*(double)(nx-kx);
    
    for (kx=0; kx<nx; kx++)	
	{
		double re=func_m[kx][0];
		double im=func_m[kx][1];
		double phase = - double(kx)*M_PI;
		func_m[kx][0] = re*cos(phase) - im*sin(phase);
		func_m[kx][1] = re*sin(phase) + im*cos(phase);
	}
    file.open("func_momentum.dat", ios::out | ios::trunc);
    for (kx=nx/2; kx<nx; kx++)
		file << px[kx] << " "<< func_m[kx][0]*dx << " "<< func_m[kx][1]*dx << endl;
	for (kx=0; kx<nx/2; kx++)
		file << px[kx] << " "<< func_m[kx][0]*dx << " "<< func_m[kx][1]*dx << endl;
	file.close();
    fftw_execute(plan_bwd); // tranfsormation in coordinate representation
    for(kx=0; kx<nx; kx++)
    {
		func_c[kx][0]=func_c[kx][0]*dpx/(2.0*M_PI);
		func_c[kx][1]=func_c[kx][1]*dpx/(2.0*M_PI);
	}
	file.open("func_coord_final.dat", ios::out | ios::trunc);
    for(kx=0; kx<nx; kx++)
    {
		x_coordinate=xmin+dx*(double)kx;
		file << x_coordinate << " "<< func_c[kx][0] << " "<< func_c[kx][1] << endl;
	}
	file.close();
    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(func_c);
    fftw_free(func_m);
    delete [] px;
    return 0;    
}
