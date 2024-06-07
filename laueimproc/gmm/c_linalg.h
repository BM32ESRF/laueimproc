/* Simple linear algebra tools. */

#define EPS 1.19209e-7
#define PY_SSIZE_T_CLEAN
#include <math.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int ObsToMeanCov(const npy_float32* roi, const npy_int16 bbox[4], npy_float32* mean, npy_float32* cov) {
    // Compute the barycenter and the biaised covariance cov [sigma1**2, sigma2**2, corr].
    long shift;
    npy_float32 weight, norm = 0.0;
    npy_float32 pos[2];

    // get mean and total intensity
    mean[0] = 0.0, mean[1] = 0.0;
    for (long i = 0; i < (long)bbox[2]; ++i) {
        shift = (long)bbox[3] * i;
        pos[0] = (npy_float)i;
        for (long j = 0; j < bbox[3]; ++j) {
            pos[1] = (npy_float)j;
            weight = roi[j + shift];
            mean[0] += pos[0] * weight, mean[1] += pos[1] * weight;  // SIMD
            norm += weight;
        }
    }
    if (!norm) {
        fprintf(stderr, "failed to compute pca because all the dup_w are equal to 0\n");
        return 1;
    }
    norm = 1 / norm;
    mean[0] *= norm, mean[1] *= norm;  // SIMD

    // get cov matrix
    cov[0] = 0.0, cov[1] = 0.0, cov[2] = 0.0;
    for (long i = 0; i < (long)bbox[2]; ++i) {
        shift = (long)bbox[3] * i;
        pos[0] = (npy_float)i - mean[0];  // i centered
        for (long j = 0; j < bbox[3]; ++j) {
            pos[1] = (npy_float)j - mean[1];  // j centered
            weight = roi[j + shift];
            cov[2] += weight * pos[0] * pos[1];
            cov[0] += weight * pos[0] * pos[0], cov[1] += weight * pos[1] * pos[1];  // SIMD
        }
    }
    cov[2] *= norm;
    cov[0] *= norm, cov[1] *= norm;

    // relative to abs base
    mean[0] += 0.5 + (npy_float32)bbox[0], mean[1] += 0.5 + (npy_float32)bbox[1];  // SIMD
    return 0;
}


int Cov2dToEigtheta(npy_float32* sigma1_std1, npy_float32* sigma2_std2, npy_float32* corr_theta) {
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
    if (*sigma1_std1 < 0.0 || *sigma2_std2 < 0.0) {
        fprintf(stderr, "the covariance matrix is not diagonalisable, (eig values < 0)\n");
        return 1;
    }
    *sigma1_std1 = sqrtf(*sigma1_std1), *sigma2_std2 = sqrtf(*sigma2_std2);  // std1, std2
    return 0;
}
