/* Very basic function for multivariate gaussian computing. */

#define EPS 1.19209e-7
#define PY_SSIZE_T_CLEAN
#include <numpy/arrayobject.h>
#include <math.h>
#include <Python.h>
#include <stdio.h>


int GMM(
    const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* mag, const long n_clu,
    npy_float32* predictions
) {
    /*
        Compute the proba of each pixel, set the result in predictions.
    */
    npy_float32* fact_const;  // contains the value independent of i
    const long area = height * width;

    // precompute const values, i loop invariant
    fact_const = malloc(2 * n_clu * sizeof(*fact_const));
    if (fact_const == NULL) {
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    #pragma omp simd safelen(1)
    for (long k = 0; k < n_clu; ++k) {
        npy_float32 sigma_d0 = cov[4 * k], corr = cov[4 * k + 1], sigma_d1 = cov[4 * k + 3];
        npy_float32 inv_det = 1.0 / (sigma_d0 * sigma_d1 - corr * corr);
        fact_const[2 * k] = inv_det;
        fact_const[2 * k + 1] = sqrtf(inv_det);
    }

    // compute proba
    for (long i = 0; i < area; ++i) {
        npy_float32 roi_pred = 0.0;  // the predicted pixel value (proba density)
        npy_float32 pos_i, pos_j;  // the positions
        pos_i = (npy_float32)(i / width + anchor_i) + 0.5, pos_j = (npy_float32)(i % width + anchor_j) + 0.5;
        #pragma omp simd reduction(+:roi_pred) safelen(1)
        for (long k = 0; k < n_clu; ++k) {
            // compute the proba
            npy_float32 mu_d0 = mean[2 * k], mu_d1 = mean[2 * k + 1];
            npy_float32 sigma_d0 = cov[4 * k], corr = cov[4 * k + 1], sigma_d1 = cov[4 * k + 3];

            // weighted gmm direct evaluation
            npy_float32 inv_det = fact_const[2 * k];
            npy_float32 sqrt_inv_det = fact_const[2 * k + 1];
            npy_float32 pos_i_cent = pos_i - mu_d0, pos_j_cent = pos_j - mu_d1;
            npy_float32 x1 = inv_det * pos_i_cent, x2 = inv_det * pos_j_cent;
            npy_float32 x3 = sigma_d0 * x2 - corr * x1, x4 = sigma_d1 * x1 - corr * x2;
            x1 = -0.5 * pos_i_cent, x2 = -0.5 * pos_j_cent;
            npy_float32 prob = expf(x4 * x1 + x3 * x2) / M_PI;
            prob *= 0.5 * mag[k] * sqrt_inv_det;

            // sum of the gaussians contribution
            roi_pred += prob;
        }
        predictions[i] = roi_pred;
    }
    free(fact_const);
    return 0;
}


int LogLikelihood(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu,
    npy_float32* log_likelihood
) {
    /*
        Compute the Log Likelihood of the gmm.
    */
    npy_float32* fact_const;
    const long area = height * width;
    npy_float32 sub_log_likelihood;

    // precompute const values, i loop invariant
    fact_const = malloc(2 * n_clu * sizeof(*fact_const));
    if (fact_const == NULL) {
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    #pragma omp simd safelen(1)
    for (long k = 0; k < n_clu; ++k) {
        npy_float32 sigma_d0 = cov[4 * k], corr = cov[4 * k + 1], sigma_d1 = cov[4 * k + 3];
        npy_float32 inv_det = 1.0 / (sigma_d0 * sigma_d1 - corr * corr);
        fact_const[2 * k] = inv_det;
        fact_const[2 * k + 1] = sqrtf(inv_det);
    }

    // compute proba
    for (long i = 0; i < area; ++i) {
        sub_log_likelihood = 0.0;
        npy_float32 pos_i = (npy_float32)(i / width + anchor_i) + 0.5, pos_j = (npy_float32)(i % width + anchor_j) + 0.5;
        #pragma omp simd reduction(+:sub_log_likelihood) safelen(1)
        for (long k = 0; k < n_clu; ++k) {
            // compute the proba
            npy_float32 mu_d0 = mean[2 * k], mu_d1 = mean[2 * k + 1];
            npy_float32 sigma_d0 = cov[4 * k], corr = cov[4 * k + 1], sigma_d1 = cov[4 * k + 3];

            // gmm direct evaluation
            npy_float32 inv_det = fact_const[2 * k];
            npy_float32 sqrt_inv_det = fact_const[2 * k + 1];
            npy_float32 pos_i_cent = pos_i - mu_d0, pos_j_cent = pos_j - mu_d1;
            npy_float32 x1 = inv_det * pos_i_cent, x2 = inv_det * pos_j_cent;
            npy_float32 x3 = sigma_d0 * x2 - corr * x1, x4 = sigma_d1 * x1 - corr * x2;
            x1 = -0.5 * pos_i_cent, x2 = -0.5 * pos_j_cent;
            npy_float32 prob = expf(x4 * x1 + x3 * x2) / M_PI;
            prob *= 0.5 * sqrt_inv_det;

            // likelihood
            prob = powf(prob, roi[i]);
            prob *= eta[k];
            sub_log_likelihood += prob;
        }
        // sum(log) more accurate than log(prod), (even if computing less efficient)
        sub_log_likelihood += EPS;  // for stability
        sub_log_likelihood = logf(sub_log_likelihood);
        *log_likelihood += sub_log_likelihood;
    }
    free(fact_const);
    return 0;
}


int MSECost(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* mag, const long n_clu,
    npy_float32* cost
) {
    /*
        Compute the mse loss of the roi and the gmm.

        rois : the pixels value of the roi
        height, width : the shape of the roi
        mean : the mean points of shape (k, 2)
        cov : the covarianc matricies of shape (k, 2, 2)
        mag : the relative weitgh of shape (k,)

        cost : the scalar loss value
    */
    npy_float32* fact_const;  // contains the value independent of i
    const long area = height * width;
    const npy_float32 inv_area = 1.0 / (npy_float32)area;

    *cost = 0.0;

    // precompute const values, i loop invariant
    fact_const = malloc(2 * n_clu * sizeof(*fact_const));
    if (fact_const == NULL) {
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    #pragma omp simd safelen(1)
    for (long k = 0; k < n_clu; ++k) {
        npy_float32 sigma_d0 = cov[4 * k], corr = cov[4 * k + 1], sigma_d1 = cov[4 * k + 3];
        npy_float32 inv_det = 1.0 / (sigma_d0 * sigma_d1 - corr * corr);
        fact_const[2 * k] = inv_det;
        fact_const[2 * k + 1] = sqrtf(inv_det);
    }

    // compute the prediction for each pixel and reduce it
    for (long i = 0; i < area; ++i) {
        npy_float32 roi_pred = 0.0;  // the predicted pixel value (proba density)
        npy_float32 pos_i, pos_j;  // the positions
        pos_i = (npy_float32)(i / width + anchor_i) + 0.5, pos_j = (npy_float32)(i % width + anchor_j) + 0.5;
        #pragma omp simd reduction(+:roi_pred) safelen(1)
        for (long k = 0; k < n_clu; ++k) {
            // compute the proba
            npy_float32 mu_d0 = mean[2 * k], mu_d1 = mean[2 * k + 1];
            npy_float32 sigma_d0 = cov[4 * k], corr = cov[4 * k + 1], sigma_d1 = cov[4 * k + 3];

            // weighted gmm direct evaluation
            npy_float32 inv_det = fact_const[2 * k];
            npy_float32 sqrt_inv_det = fact_const[2 * k + 1];
            npy_float32 pos_i_cent = pos_i - mu_d0, pos_j_cent = pos_j - mu_d1;
            npy_float32 x1 = inv_det * pos_i_cent, x2 = inv_det * pos_j_cent;
            npy_float32 x3 = sigma_d0 * x2 - corr * x1, x4 = sigma_d1 * x1 - corr * x2;
            x1 = -0.5 * pos_i_cent, x2 = -0.5 * pos_j_cent;
            npy_float32 prob = expf(x4 * x1 + x3 * x2) / M_PI;
            prob *= 0.5 * mag[k] * sqrt_inv_det;

            // sum of the gaussians contribution
            roi_pred += prob;
        }
        // mse loss
        npy_float32 buff = roi_pred - roi[i];
        *cost += buff * buff;
    }
    free(fact_const);
    *cost *= inv_area;  // average mse loss
    return 0;
}


int MSECostAndGrad(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* mag, const long n_clu,
    npy_float32* cost, npy_float32* mean_grad, npy_float32* cov_grad, npy_float32* mag_grad
) {
    /*
        Compute the mse loss of the roi and the gmm. Compute the gradiant as well.

        rois : the pixels value of the roi
        height, width : the shape of the roi
        mean : the mean points of shape (k, 2)
        cov : the covarianc matricies of shape (k, 2, 2)
        mag : the relative weitgh of shape (k,)

        cost : the scalar loss value
        mean_grad : the gradient of mean of shape (k, 2)
        cov_grad : the gradient of cov of shape (k, 2, 2)
        mag_grad : the gradient of mag of shape (k,)
    */
    npy_float32* fact_const;  // contains the value independent of i
    const long area = height * width;
    const npy_float32 inv_area = 1.0 / (npy_float32)area;

    // set values to zero for sumation
    *cost = 0.0;
    memset(mean_grad, 0, n_clu * 2 * sizeof(*mean_grad));
    memset(cov_grad, 0, n_clu * 4 * sizeof(*cov_grad));
    memset(mag_grad, 0, n_clu * sizeof(*mag_grad));

    // precompute const values, i loop invariant
    fact_const = malloc(7 * n_clu * sizeof(*fact_const));
    if (fact_const == NULL) {
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    #pragma omp simd safelen(1)
    for (long k = 0; k < n_clu; ++k) {
        npy_float32 sigma_d0 = cov[4 * k], corr = cov[4 * k + 1], sigma_d1 = cov[4 * k + 3];
        npy_float32 diag = sigma_d0 * sigma_d1;
        npy_float32 c2 = corr * corr;
        npy_float32 inv_det = 1.0 / (diag - c2);
        fact_const[7 * k] = -diag;
        fact_const[7 * k + 1] = -c2;
        fact_const[7 * k + 2] = inv_det;
        fact_const[7 * k + 3] = inv_det * inv_det;
        fact_const[7 * k + 4] = sqrtf(inv_det);
        fact_const[7 * k + 5] = sigma_d0 * sigma_d0;
        fact_const[7 * k + 6] = sigma_d1 * sigma_d1;
    }

    // compute jacobian and reduce it as grad
    for (long i = 0; i < area; ++i) {
        npy_float32 roi_pred = 0.0;  // the predicted pixel value (proba density)
        npy_float32 pos_i, pos_j;  // the positions

        pos_i = (npy_float32)(i / width + anchor_i) + 0.5, pos_j = (npy_float32)(i % width + anchor_j) + 0.5;

        #pragma omp simd reduction(+:roi_pred) safelen(1)
        for (long k = 0; k < n_clu; ++k) {
            // compute the proba
            npy_float32 mu_d0 = mean[2 * k], mu_d1 = mean[2 * k + 1];
            npy_float32 sigma_d0 = cov[4 * k], corr = cov[4 * k + 1], sigma_d1 = cov[4 * k + 3];

            // weighted gmm direct evaluation
            npy_float32 inv_det = fact_const[7 * k + 2];
            npy_float32 sqrt_inv_det = fact_const[7 * k + 4];
            npy_float32 pos_i_cent = pos_i - mu_d0, pos_j_cent = pos_j - mu_d1;
            npy_float32 x1 = inv_det * pos_i_cent, x2 = inv_det * pos_j_cent;
            npy_float32 x3 = sigma_d0 * x2 - corr * x1, x4 = sigma_d1 * x1 - corr * x2;
            x1 = -0.5 * pos_i_cent, x2 = -0.5 * pos_j_cent;
            npy_float32 prob = expf(x4 * x1 + x3 * x2) / M_PI;
            prob *= 0.5 * mag[k] * sqrt_inv_det;

            // sum of the gaussians contribution
            roi_pred += prob;
        }
        for (long k = 0; k < n_clu; ++k) {
            // compute the jacobians
            npy_float32 mean_jac_i, mean_jac_j;  // (N, K, 2)
            npy_float32 cov_jac_s1, cov_jac_s2, cov_jac_c;  // (N, K, 2, 2)
            npy_float32 mag_jac;  // (N, K)

            npy_float32 mu_d0 = mean[2 * k], mu_d1 = mean[2 * k + 1];
            npy_float32 sigma_d0 = cov[4 * k], corr = cov[4 * k + 1], sigma_d1 = cov[4 * k + 3];
            npy_float32 nu = mag[k];

            npy_float32 mdiag = fact_const[7 * k];
            npy_float32 mc2 = fact_const[7 * k + 1];
            npy_float32 inv_det = fact_const[7 * k + 2];
            npy_float32 inv_det_2 = fact_const[7 * k + 3];
            npy_float32 sqrt_inv_det = fact_const[7 * k + 4];

            npy_float32 pos_i_cent = pos_i - mu_d0, pos_j_cent = pos_j - mu_d1;
            cov_jac_c = -inv_det * pos_j_cent;
            npy_float32 _9 = -inv_det * pos_i_cent;
            mean_jac_i = cov_jac_c * corr;
            mean_jac_j = -_9 * sigma_d1;
            mean_jac_i = mean_jac_i + mean_jac_j;
            mean_jac_j = _9 * corr;
            cov_jac_s2 = -cov_jac_c * sigma_d0;
            mean_jac_j = mean_jac_j + cov_jac_s2;
            cov_jac_s2 = -0.5 * mean_jac_j * pos_j_cent;
            npy_float32 _13 = -0.5 * mean_jac_i * pos_i_cent;
            cov_jac_s2 = cov_jac_s2 + _13;
            cov_jac_s2 = expf(cov_jac_s2);
            _13 = (0.5 / M_PI) * cov_jac_s2;
            mag_jac = _13 * sqrt_inv_det;
            npy_float32 _15 = mag_jac * nu;
            npy_float32 _2 = inv_det;
            npy_float32 _3 = sqrt_inv_det * nu;
            _2 *= _3;
            npy_float32 _16 = (-0.25 / M_PI) * cov_jac_s2 * _2;
            npy_float32 _17 = inv_det_2 * pos_i_cent, _18 = inv_det_2 * pos_j_cent;
            npy_float32 _4 = corr * sigma_d1;
            npy_float32 _19 = -0.5 * pos_i_cent, _20 = -0.5 * pos_j_cent;
            npy_float32 _21 = corr * sigma_d0;
            mean_jac_i *= _15, mean_jac_j *= _15;
            cov_jac_s1 = _16 * sigma_d1;
            npy_float32 _23 = _18 * _4;
            _23 -= _17 * fact_const[7 * k + 6];
            _23 *= _19;
            npy_float32 _25 = -cov_jac_c;
            pos_i_cent *= inv_det_2 *  corr * sigma_d1;
            _25 += _18 * mdiag + pos_i_cent;
            _25 *= _20;
            _23 += _25;
            _3 *= 0.5 / M_PI;
            _23 *= cov_jac_s2 * _3;
            cov_jac_s1 += _23;
            _16 *= sigma_d0;
            _23 = -_9;
            _25 = _17 * mdiag;
            _23 += _25 + inv_det_2 * pos_j_cent * corr * sigma_d0;
            _23 *= _19;
            _25 = _17 * _21;
            _25 -= fact_const[7 * k + 5] * _18;
            _25 *= _20;
            _23 += _25;
            cov_jac_s2 *= _23 * _3;
            cov_jac_s2 += _16;
            cov_jac_c += 2.0 * (mc2 * _18 + _17 * _4);
            cov_jac_c *= _19;
            _9 += 2.0 * (mc2 * _17 + _18 * _21);
            _9 *= _20;
            cov_jac_c += _9;
            cov_jac_c *= _15;
            cov_jac_c += _13 * _2 * corr;

            // reduce jacobians to grad
            npy_float32 mse_grad = 2 * (roi_pred - roi[i]) * inv_area;

            mean_jac_i *= mse_grad, mean_jac_j *= mse_grad;
            mean_grad[2 * k] += mean_jac_i;
            mean_grad[2 * k + 1] += mean_jac_j;

            cov_jac_s1 *= mse_grad, cov_jac_s2 *= mse_grad, cov_jac_c *= mse_grad;
            cov_grad[4 * k] += cov_jac_s1;
            cov_grad[4 * k + 1] += cov_jac_c;  // copy cov_grad[4 * k + 2] later
            cov_grad[4 * k + 3] += cov_jac_s2;

            mag_jac *= mse_grad;
            mag_grad[k] += mag_jac;
        }
        // mse loss
        npy_float32 buff = roi_pred - roi[i];
        *cost += buff * buff;

        // cov symetric
        for (long k = 0; k < n_clu; ++k) {
            cov_grad[4 * k + 2] = cov_grad[4 * k + 1];
        }
    }
    free(fact_const);
    *cost *= inv_area;  // average mse loss
    return 0;
}
