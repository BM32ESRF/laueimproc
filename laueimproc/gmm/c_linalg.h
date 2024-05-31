/* Simple linear algebra tools. */

#define PY_SSIZE_T_CLEAN
#include <math.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


#define EPS 1.1920929e-7


void Cov2dToEigtheta(npy_float32* sigma1_std1, npy_float32* sigma2_std2, npy_float32* corr_theta) {
    // Diagonalise inplace a covariance matrix with a rotation and a digonal matrix.
    npy_float32 buff1, buff2;
    *corr_theta *= 2.0;  // 2*c
    buff1 = (*sigma1_std1) + (*sigma2_std2);  // s1+s2
    *sigma2_std2 -= *sigma1_std1;  // s2-s1
    *sigma1_std1 = (*corr_theta)*(*corr_theta) + (*sigma2_std2)*(*sigma2_std2);  // (2*c)**2 + (s2-s1)**2
    *sigma1_std1 = sqrtf(*sigma1_std1);  // sqrt((2*c)**2 + (s2-s1)**2)
    if (fabsf(*corr_theta) > EPS) {  // abs(2*c) > eps
        buff2 = (*sigma1_std1) + (*sigma2_std2);  // s2 - s1 + sqrt((2*c)**2 + (s2-s1)**2)
        buff2 /= *corr_theta;  // (s2 - s1 + sqrt((2*c)**2 + (s2-s1)**2)) / (2*c)
        *corr_theta = atanf(buff2);  // theta
    }
    else {
        *corr_theta = *sigma2_std2 > EPS ? 0.5*M_PI : 0; // theta = pi/2 if s2-s1 > eps else 0
    }
    *sigma2_std2 = buff1 - (*sigma1_std1);  // s1 + s2 - sqrt((2*c)**2 + (s2-s1)**2)
    *sigma1_std1 += buff1;  // s1 + s2 + sqrt((2*c)**2 + (s2-s1)**2)
    *sigma1_std1 *= 0.5, *sigma2_std2 *= 0.5;  // lambda_1, lambda_2
    *sigma1_std1 = sqrtf(*sigma1_std1), *sigma2_std2 = sqrtf(*sigma2_std2);  // std1, std2
}
