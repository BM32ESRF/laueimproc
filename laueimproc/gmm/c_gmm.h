/* Very basic function for multivariate gaussian computing. */

#define PY_SSIZE_T_CLEAN
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int CostAndGrad(
    npy_float32* roi, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu,
    npy_float32* cost, npy_float32* mean_grad, npy_float32* cov_grad, npy_float32* eta_grad
) {
    /*
        Compute the mse loss of the roi and the gmm. Compute the gradiant as well.

        rois : the pixels value of the roi
        height, width : the shape of the roi
        mean : the mean points of shape (k, 2)
        cov : the covarianc matricies of shape (k, 2, 2)
        eta : the relative weitgh of shape (k,)

        cost : the scalar loss value
        mean_grad : the gradient of mean of shape (k, 2)
        cov_grad : the gradient of cov of shape (k, 2, 2)
        eta_grad : the gradient of eta of shape (k,)
    */
    const long area = height * width;

    *cost = 0.0;

    for (long i = 0; i < area; ++i) {
        npy_float32 roi_pred = 0.0;  // the predicted pixel value (proba density)
        npy_float32 buff;  // tmp var
        npy_float32 pos_i, pos_j;  // the positions

        pos_i = (npy_float32)(i / width) + 0.5, pos_j = (npy_float32)(i % width) + 0.5;

        for (long k = 0; k < n_clu; ++k) {
            // compute the proba
            roi_pred += 1.0;  // sum of the gaussians contribution
        }
        for (long k = 0; k < n_clu; ++k) {
            // compute the jacobians
            npy_float32 mean_jac_i = 0.0, mean_jac_j = 0.0;  // (N, K, 2)
            npy_float32 cov_jac_s1 = 0.0, cov_jac_s2 = 0.0, cov_jac_c = 0.0;  // (N, K, 2, 2)

            // reduce jacobians to grad
            mean_jac_i *= roi_pred, mean_jac_j *= roi_pred;
            mean_grad[2 * k] += mean_jac_i;
            mean_grad[2 * k + 1] += mean_jac_j;

            cov_jac_s1 *= roi_pred, cov_jac_s2 *= roi_pred, cov_jac_c *= roi_pred;
            cov_grad[4 * k] += cov_jac_s1;
            cov_grad[4 * k + 1] += cov_jac_c;  // penser a copier a la fin
            cov_grad[4 * k + 2] += cov_jac_s2;

        }

        // mse loss
        buff = roi_pred - roi[i];
        *cost += buff * buff;

        // jacobian to grad
        // roi_pred (\(N\),)
        // mean_jac (\(N\), \(K\), 2)
        // mean_grad (K, 2)
        mean_grad = sum(mean_jac * roi_pred); // somme selon n_obs axis
    }

    *cost /= (npy_float32)(area);  // average mse loss


    // mean_grad = sum(mean_jac * roi_pred); // somme selon n_obs axis
    // cov_grad = sum(cov_jac * roi_pred);
    // eta_grad = sum(eta_jac * roi_pred);

    return 0;
}

void lambdify_float(const npy_intp _dim, float *_10, float *_11, float *_12, float *_14, float *_15, float *_22, float *_8, float *corr, float *eta, float *mu_d0, float *mu_d1, float *o_d0, float *o_d1, float *sigma_d0, float *sigma_d1) {
  npy_intp _i;
  #pragma omp parallel for simd schedule(static)
  for ( _i = 0; _i < _dim; ++_i ) {
    float _0, _1, _13, _16, _17, _18, _19, _2, _20, _21, _23, _24, _25, _26, _3, _4, _5, _6, _7, _9, _buf;
    _0 = corr[_i] * corr[_i];
    _1 = -1.0f * sigma_d0[_i];
    _1 *= sigma_d1[_i];
    _1 += _0;
    _2 = -1.0f * _1;
    _3 = 1.0f / sqrtf(_2);
    _4 = -1.0f * mu_d0[_i];
    _5 = _4 + o_d0[_i];
    _4 = 1.0f / _1;
    _6 = -1.0f * mu_d1[_i];
    _7 = _6 + o_d1[_i];
    _8[_i] = _4 * _7;
    _9 = _4 * _5;
    _10[_i] = _8[_i] * corr[_i];
    _11[_i] = -1.0f * _9;
    _11[_i] *= sigma_d1[_i];
    _10[_i] += _11[_i];
    _11[_i] = _9 * corr[_i];
    _12[_i] = -1.0f * _8[_i];
    _12[_i] *= sigma_d0[_i];
    _11[_i] += _12[_i];
    _12[_i] = -0.5f * _11[_i];
    _12[_i] *= _7;
    _13 = -0.5f * _10[_i];
    _13 *= _5;
    _12[_i] += _13;
    _12[_i] = expf(_12[_i]);
    _13 = 0.15915494309189535f * _12[_i];
    _14[_i] = _13 * _3;
    _15[_i] = _14[_i] * eta[_i];
    _buf = _2;
    _2 *= _2;
    _2 *= _buf;
    _2 = 1.0f / sqrtf(_2);
    _2 *= eta[_i];
    _16 = 0.07957747154594767f * _12[_i];
    _16 *= _2;
    _1 *= _1;
    _1 = 1.0f/_1;
    _17 = _1 * _5;
    _18 = _1 * _7;
    _4 = corr[_i] * sigma_d1[_i];
    _19 = 0.5f * _5;
    _6 = sigma_d0[_i] * sigma_d1[_i];
    _20 = 0.5f * _7;
    _21 = corr[_i] * sigma_d0[_i];
    _0 *= 2.0f;
    _10[_i] *= _15[_i];
    _11[_i] *= _15[_i];
    _22[_i] = -1.0f * _16;
    _22[_i] *= sigma_d1[_i];
    _23 = _18 * _4;
    _24 = sigma_d1[_i] * sigma_d1[_i];
    _25 = -1.0f * _17;
    _25 *= _24;
    _23 += _25;
    _23 *= -1.0f;
    _23 *= _19;
    _25 = -1.0f * _8[_i];
    _26 = -1.0f * _18;
    _26 *= _6;
    _5 *= _1;
    _5 *= corr[_i];
    _5 *= sigma_d1[_i];
    _25 += _26;
    _25 += _5;
    _25 *= -1.0f;
    _25 *= _20;
    _23 += _25;
    _23 *= 0.15915494309189535f;
    _23 *= _12[_i];
    _23 *= _3;
    _23 *= eta[_i];
    _22[_i] += _23;
    _16 *= -1.0f;
    _16 *= sigma_d0[_i];
    _23 = -1.0f * _9;
    _25 = -1.0f * _17;
    _25 *= _6;
    _26 = _1 * _7;
    _26 *= corr[_i];
    _26 *= sigma_d0[_i];
    _23 += _25;
    _23 += _26;
    _23 *= -1.0f;
    _23 *= _19;
    _25 = _17 * _21;
    _1 = sigma_d0[_i] * sigma_d0[_i];
    _26 = -1.0f * _1;
    _26 *= _18;
    _25 += _26;
    _25 *= -1.0f;
    _25 *= _20;
    _23 += _25;
    _12[_i] *= 0.15915494309189535f;
    _12[_i] *= _23;
    _12[_i] *= _3;
    _12[_i] *= eta[_i];
    _12[_i] += _16;
    _16 = -1.0f * _0;
    _16 *= _18;
    _23 = 2.0f * _17;
    _23 *= _4;
    _8[_i] += _16;
    _8[_i] += _23;
    _8[_i] *= -1.0f;
    _8[_i] *= _19;
    _16 = -1.0f * _0;
    _16 *= _17;
    _17 = 2.0f * _18;
    _17 *= _21;
    _9 += _16;
    _9 += _17;
    _9 *= -1.0f;
    _9 *= _20;
    _8[_i] += _9;
    _8[_i] *= _15[_i];
    _9 = _13 * _2;
    _9 *= corr[_i];
    _8[_i] += _9;
  }
}

def lambdify(corr, eta, mu_d0, mu_d1, o_d0, o_d1, sigma_d0, sigma_d1):
    // this section is not cached and not compiled
    _0 = corr**2
    _1 = -sigma_d0*sigma_d1
    _1 = _0 + _1
    _2 = -_1
    _3 = _2**(-0.5)
    _4 = -mu_d0
    _5 = _4 + o_d0
    _4 = 1/_1
    _6 = -mu_d1
    _7 = _6 + o_d1
    _8 = _4*_7
    _9 = _4*_5
    _10 = _8*corr
    _11 = -_9*sigma_d1
    _10 = _10 + _11
    _11 = _9*corr
    _12 = -_8*sigma_d0
    _11 = _11 + _12
    _12 = -_11*_7/2
    _13 = -_10*_5/2
    _12 = _12 + _13
    _12 = exp(_12)
    _13 = 0.1591549430918953357688838*_12
    _14 = _13*_3
    _15 = _14*eta
    _2 = _2**(-1.5)
    _2 = _2*eta
    _16 = 0.07957747154594766788444188*_12*_2
    _1 = _1**(-2)
    _17 = _1*_5
    _18 = _1*_7
    _4 = corr*sigma_d1
    _19 = 0.5*_5
    _6 = sigma_d0*sigma_d1
    _20 = 0.5*_7
    _21 = corr*sigma_d0
    _0 = 2.0*_0
    _10 = _10*_15
    _11 = _11*_15
    _22 = -_16*sigma_d1
    _23 = _18*_4
    _24 = sigma_d1**2
    _25 = -_17*_24
    _23 = _23 + _25
    _23 = -_19*_23
    _25 = -_8
    _26 = -_18*_6
    _5 = _1*_5*corr*sigma_d1
    _25 = _25 + _26 + _5
    _25 = -_20*_25
    _23 = _23 + _25
    _23 = 0.1591549430918953357688838*_12*_23*_3*eta
    _22 = _22 + _23
    _16 = -_16*sigma_d0
    _23 = -_9
    _25 = -_17*_6
    _26 = _1*_7*corr*sigma_d0
    _23 = _23 + _25 + _26
    _23 = -_19*_23
    _25 = _17*_21
    _1 = sigma_d0**2
    _26 = -_1*_18
    _25 = _25 + _26
    _25 = -_20*_25
    _23 = _23 + _25
    _12 = 0.1591549430918953357688838*_12*_23*_3*eta
    _12 = _12 + _16
    _16 = -_0*_18
    _23 = 2.0*_17*_4
    _8 = _16 + _23 + _8
    _8 = -_19*_8
    _16 = -_0*_17
    _17 = 2.0*_18*_21
    _9 = _16 + _17 + _9
    _9 = -_20*_9
    _8 = _8 + _9
    _8 = _15*_8
    _9 = _13*_2*corr
    _8 = _8 + _9
    return [_15, _14, _10, _11, _22, _12, _8]