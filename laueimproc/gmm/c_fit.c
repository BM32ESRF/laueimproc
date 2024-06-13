/* Regression algorithm to fit 2d multivariate gaussians mixture model. */

#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include "laueimproc/gmm/c_gmm.h"
#include "laueimproc/gmm/c_linalg.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>


typedef struct point {
    npy_float32 x;
    npy_float32 y;
} point_t;


point_t RandomNormal() {
    // Draw two samples -> N(0, 1).
    // srand(time(NULL));  // stdlib is assumed to be already called
    point_t draw;
    npy_float32 u, v;
    do {
        draw.x = (2.0 * (npy_float32)rand() / (npy_float32)RAND_MAX) - 1.0;  // stdlib.h
        draw.y = (2.0 * (npy_float32)rand() / (npy_float32)RAND_MAX) - 1.0;
        u = draw.x * draw.x + draw.y * draw.y;
    } while (u >= 1.0 || u == 0.0);
    v = logf(u);
    v *= -2.0;
    v /= u;
    v = sqrtf(v);
    draw.x *= v, draw.y *= v;
    return draw;
}


npy_float32 MaxAbsDiff(const npy_float32* x, const npy_float32* y, const long n) {
    npy_float32 diff = fabsf(x[0] - y[0]), diff_;
    for (long i = 1; i < n; ++i) {
        diff_ = fabsf(x[i] - y[i]);
        if (diff_ > diff) {
            diff = diff_;
        }
    }
    return diff;
}


int FitInit(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu
) {
    /* Draw the random initial conditions. */
    npy_int16 bbox[4] = {anchor_i, anchor_j, height, width};
    npy_float32 avg_mean[2], std1_std2_theta[3];
    point_t point;
    npy_float32 cos_theta, sin_theta;

    // init cov
    if (ObsToMeanCov(roi, bbox, avg_mean, std1_std2_theta)) {
        return 1;
    }
    for (long j = 0; j < n_clu; ++j) {
        npy_float32 inv_nclu2 = 1.0 / (n_clu * n_clu);
        cov[4 * j] = std1_std2_theta[0] * inv_nclu2;
        cov[4 * j + 1] = std1_std2_theta[2] * inv_nclu2;
        cov[4 * j + 2] = std1_std2_theta[2] * inv_nclu2;
        cov[4 * j + 3] = std1_std2_theta[1] * inv_nclu2;
    }

    // init mean
    if (Cov2dToEigtheta(&(std1_std2_theta[0]), &(std1_std2_theta[1]), &(std1_std2_theta[2]))) {
        return 1;
    }
    cos_theta = cosf(std1_std2_theta[2]), sin_theta = sinf(std1_std2_theta[2]);
    for (long j = 0; j < n_clu; ++j) {
        point = RandomNormal();
        mean[2 * j] = (point.x * std1_std2_theta[0] * cos_theta) - (point.y * std1_std2_theta[1] * sin_theta);
        mean[2 * j + 1] = (point.x * std1_std2_theta[0] * sin_theta) + (point.y * std1_std2_theta[1] * cos_theta);
        mean[2 * j] += avg_mean[0];
        mean[2 * j + 1] += avg_mean[1];
    }

    // init eta
    for (long j = 0; j < n_clu; ++j) {
        eta[j] = 1.0 / (npy_float32)n_clu;
    }

    return 0;
}


int FitEMOneStep(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu
) {
    /*
        Implements one step of the Espectation Maximization algorithm.
    */
    const long area = height * width;
    npy_float32* post;
    npy_float32 norm, sum;
    long shift;

    // posterior probability that observation i belongs to cluster j.
    post = malloc(n_clu * area * sizeof(*post));
    if (post == NULL) {
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    for (long j = 0; j < n_clu; ++j) {  // compute each gaussian separately
        GMM(
            anchor_i, anchor_j, height, width,
            mean + 2 * j, cov + 4 * j, eta + j, 1,
            post + j * area
        );
    }
    for (long i = 0; i < area; ++i) {  // normalize contributions
        norm = 0.0;
        #pragma omp simd reduction(+:norm) safelen(1)
        for (long j = 0; j < n_clu; ++j) {
            post[i + j * area] += EPS;  // against div by 0 and more stability
            norm += post[i + j * area];
        }
        norm = 1.0 / norm;
        #pragma omp simd safelen(1)
        for (long j = 0; j < n_clu; ++j) {
            post[i + j * area] *= norm;
        }
    }

    // posterior -> alpha_i * p_ij
    #pragma omp simd collapse(2)
    for (long j = 0; j < n_clu; ++j) {
        for (long i = 0; i < area; ++i) {
            post[i + j * area] *= roi[i];
        }
    }

    // the relative weight of each gaussian
    norm = 0.0;
    #pragma omp simd reduction(+:norm)
    for (long i = 0; i < area; ++i) {
        norm += roi[i];
    }
    norm = 1.0 / norm;
    for (long j = 0; j < n_clu; ++j) {
        shift = j * area;
        sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (long i = 0; i < area; ++i) {
            sum += post[i + shift];
        }
        eta[j] = sum * norm;

        // the mean of each gaussian
        mean[2 * j] = 0.0;
        mean[2 * j + 1] = 0.0;
        for (long i = 0; i < area; ++i) {
            npy_float32 pos_i = (npy_float32)(i / width), pos_j = (npy_float32)(i % width);
            mean[2 * j] += post[i + shift] * pos_i;
            mean[2 * j + 1] += post[i + shift] * pos_j;
        }
        mean[2 * j] /= sum;
        mean[2 * j + 1] /= sum;
        mean[2 * j] += (npy_float32)anchor_i + 0.5;
        mean[2 * j + 1] += (npy_float32)anchor_j + 0.5;

        // the cov of each gaussian
        cov[4 * j] = 0.0;
        cov[4 * j + 1] = 0.0;
        cov[4 * j + 3] = 0.0;
        for (long i = 0; i < area; ++i) {
            npy_float32 pos_i = (npy_float32)(i / width), pos_j = (npy_float32)(i % width);
            pos_i += (npy_float32)anchor_i + 0.5, pos_j += (npy_float32)anchor_j + 0.5;
            pos_i -= mean[2 * j], pos_j -= mean[2 * j + 1];

            npy_float32 weight = post[i + shift];
            cov[4 * j] += weight * pos_i * pos_i;
            cov[4 * j + 1] += weight * pos_i * pos_j;
            cov[4 * j + 3] += weight * pos_j * pos_j;
        }
        cov[4 * j] /= sum;
        cov[4 * j + 1] /= sum;
        cov[4 * j + 2] = cov[4 * j + 1];
        cov[4 * j + 3] /= sum;
    }

    free(post);
    return 0;
}


int FitEMMultiSteps(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu
) {
    /*
        Implement the Espectation Maximization algorithm.
        Assume that the Gaussians parameters have already been correctly initialized.
        Performs a single training, if serveral tries must be pefrormed, it is done above.
    */
    npy_float32* prev_mean;
    prev_mean = malloc(2 * n_clu * sizeof(*prev_mean));
    if (prev_mean == NULL) {
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    do {
        memcpy(prev_mean, mean, 2 * n_clu * sizeof(*prev_mean));
        if (
            FitEMOneStep(
                roi, anchor_i, anchor_j, height, width,
                mean, cov, eta, n_clu
            )
        ) {
            free(prev_mean);
            return 1;
        }
    } while (MaxAbsDiff(mean, prev_mean, 2 * n_clu) > 0.01);
    free(prev_mean);
    return 0;
}


int GradStep(
    npy_float32* ref_mean, npy_float32* ref_cov, npy_float32* ref_mag,
    npy_float32* mean_grad, npy_float32* cov_grad, npy_float32* mag_grad,
    npy_float32 lr, const long n_clu,
    npy_float32* mean, npy_float32* cov, npy_float32* mag
) {
    #pragma omp simd safelen(1)
    for (long k = 0; k < n_clu; ++k) {
        mean[2 * k] = ref_mean[2 * k] - lr * mean_grad[2 * k];
        mean[2 * k + 1] = ref_mean[2 * k + 1] - lr * mean_grad[2 * k + 1];
        cov[4 * k] = ref_cov[4 * k] - lr * cov_grad[4 * k];
        cov[4 * k + 1] = ref_cov[4 * k + 1] - lr * cov_grad[4 * k + 1];
        cov[4 * k + 2] = ref_cov[4 * k + 2] - lr * cov_grad[4 * k + 2];
        cov[4 * k + 3] = ref_cov[4 * k + 3] - lr * cov_grad[4 * k + 3];
        mag[k] = ref_mag[k] - lr * mag_grad[k];
    }
    return 0;
}


int FitMSEOptimalStep(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* mag, const long n_clu,
    npy_float32* lr, npy_float32* cost, npy_float32* mean_grad, npy_float32* cov_grad, npy_float32* mag_grad
) {
    /*
        Find the step that minimise the loss in the oposit direction of the gradient.
        Performs a dichotomy search.
    */
    npy_float32 left_lr = 0.0, right_lr = *lr * 8.0;
    npy_float32 left_cost = *cost, right_cost;
    npy_float32 *initial_mean, *initial_cov, *initial_mag;

    // find right boundary
    long count = 0;
    for (long k = 0; k < n_clu; ++k) {
        while (  // det(cov) <= 0
            (cov[4 * k] - right_lr * cov_grad[4 * k]) * (cov[4 * k + 3] - right_lr * cov_grad[4 * k + 3])
            - (cov_grad[4 * k + 1] - right_lr * cov_grad[4 * k + 1]) * (cov[4 * k + 2] - right_lr * cov_grad[4 * k + 2])
            <= 0
        ) {
            right_lr *= 0.5;
            if (count++ == 16) {  // failed to converge
                fprintf(stderr, "the det of covariance matrix is <= 0\n");
                *lr = left_lr;
                return 0;
            }
        }
    }

    // alloc for ref
    initial_mean = malloc(2 * n_clu * sizeof(*initial_mean));
    if (initial_mean == NULL) {
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    initial_cov = malloc(4 * n_clu * sizeof(*initial_cov));
    if (initial_cov == NULL) {
        free(initial_mean);
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    initial_mag = malloc(n_clu * sizeof(*initial_mag));
    if (initial_mag == NULL) {
        free(initial_mean);
        free(initial_cov);
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    memcpy(initial_mean, mean, 2 * n_clu * sizeof(*mean));
    memcpy(initial_cov, cov, 4 * n_clu * sizeof(*cov));
    memcpy(initial_mag, mag, n_clu * sizeof(*mag));

    // compute border costs
    GradStep(
        initial_mean, initial_cov, initial_mag,
        mean_grad, cov_grad, mag_grad,
        right_lr, n_clu,
        mean, cov, mag
    );
    if (
        MSECost(
            roi, anchor_i, anchor_j, height, width,
            mean, cov, mag, n_clu,
            &right_cost
        )
    ) {
        free(initial_mean);
        free(initial_cov);
        free(initial_mag);
        return 1;
    }

    // dichotomy
    for (long i = 0; i < 16; ++i) {
        *lr = 0.5 * (left_lr + right_lr);
        GradStep(
            initial_mean, initial_cov, initial_mag,
            mean_grad, cov_grad, mag_grad,
            *lr, n_clu,
            mean, cov, mag
        );
        if (
            MSECost(
                roi, anchor_i, anchor_j, height, width,
                mean, cov, mag, n_clu,
                cost
            )
        ) {
            free(initial_mean);
            free(initial_cov);
            free(initial_mag);
            return 1;
        }

        if (*cost <= left_cost) {
            left_cost = *cost;
            left_lr = *lr;
        } else {
            right_cost = *cost;
            right_lr = *lr;
        }
        // fprintf(stdout, "        lr=%f\n", *lr);
    }

    free(initial_mean);
    free(initial_cov);
    free(initial_mag);
    return 0;
}


int FitMSEMultiSteps(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* mag, const long n_clu
) {
    /*
        Minimise the mean square error between a 2d gaussian mixture model and an image.
        Assume that the Gaussians parameters have already been correctly initialized.
        Performs a single training, if serveral tries must be pefrormed, it is done above.
    */
    long i = 0;
    npy_float32 cost, lr = 1.0;
    npy_float32 *mean_grad, *cov_grad, *mag_grad, *prev_mean;

    mean_grad = malloc(2 * n_clu * sizeof(*mean_grad));
    if (mean_grad == NULL) {
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    cov_grad = malloc(4 * n_clu * sizeof(*cov_grad));
    if (cov_grad == NULL) {
        free(mean_grad);
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    mag_grad = malloc(n_clu * sizeof(*mag_grad));
    if (mag_grad == NULL) {
        free(mean_grad);
        free(cov_grad);
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }

    prev_mean = malloc(2 * n_clu * sizeof(*prev_mean));
    if (prev_mean == NULL) {
        free(mean_grad);
        free(cov_grad);
        free(mag_grad);
        fprintf(stderr, "failed to malloc\n");
        return 1;
    }
    // fprintf(stdout, "new training...\n");
    do {
        memcpy(prev_mean, mean, 2 * n_clu * sizeof(*prev_mean));
        if (  // find new direction
            MSECostAndGrad(
                roi, anchor_i, anchor_j, height, width,
                mean, cov, mag, n_clu,
                &cost, mean_grad, cov_grad, mag_grad
            )
            ||
            FitMSEOptimalStep(
                roi, anchor_i, anchor_j, height, width,
                mean, cov, mag, n_clu,
                &lr, &cost, mean_grad, cov_grad, mag_grad
            )
        ) {
            free(mean_grad);
            free(cov_grad);
            free(mag_grad);
            free(prev_mean);
            return 1;
        }
        // fprintf(stdout, "    mse=%f, lr=%f\n", cost, lr);
        if (i > 2048) {
            fprintf(stderr, "too slow 2d gmm training, exit prematurely\n");
            break;
        }
    } while (i++ < 32 || MaxAbsDiff(mean, prev_mean, 2 * n_clu) > 0.01);
    free(mean_grad);
    free(cov_grad);
    free(mag_grad);
    free(prev_mean);
    return 0;
}


int FitEM(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu, const long n_tries
) {
    /* Implement the Espectation Maximization algorithm. */
    if (
        FitInit(
            roi, anchor_i, anchor_j, height, width,
            mean, cov, eta, n_clu
        )
        ||
        FitEMMultiSteps(
            roi, anchor_i, anchor_j, height, width,
            mean, cov, eta, n_clu
        )
    ) {
        return 1;
    }
    if (n_tries > 1) {
        npy_float32 best_llh, llh;
        if (
            LogLikelihood(
                roi, anchor_i, anchor_j, height, width,
                mean, cov, eta, n_clu,
                &best_llh
            )
        ) {
            return 1;
        }
        npy_float32 *curr_mean, *curr_cov, *curr_eta;
        curr_mean = malloc(2 * n_clu * sizeof(*curr_mean));
        if (curr_mean == NULL) {
            fprintf(stderr, "failed to malloc\n");
            return 1;
        }
        curr_cov = malloc(4 * n_clu * sizeof(*curr_cov));
        if (curr_cov == NULL) {
            free(curr_mean);
            fprintf(stderr, "failed to malloc\n");
            return 1;
        }
        curr_eta = malloc(n_clu * sizeof(*curr_eta));
        if (curr_eta == NULL) {
            free(curr_mean);
            free(curr_cov);
            fprintf(stderr, "failed to malloc\n");
            return 1;
        }
        for (long i = 1; i < n_tries; ++i) {
            if (
                FitInit(
                    roi, anchor_i, anchor_j, height, width,
                    curr_mean, curr_cov, curr_eta, n_clu
                )
                ||
                FitEMMultiSteps(
                    roi, anchor_i, anchor_j, height, width,
                    curr_mean, curr_cov, curr_eta, n_clu
                )
                ||
                LogLikelihood(
                    roi, anchor_i, anchor_j, height, width,
                    curr_mean, curr_cov, curr_eta, n_clu,
                    &llh
                )
            ) {
                free(curr_mean);
                free(curr_cov);
                free(curr_eta);
                return 1;
            }
            if (llh > best_llh) {
                best_llh = llh;
                memcpy(mean, curr_mean, 2 * n_clu * sizeof(*curr_mean));
                memcpy(cov, curr_cov, 4 * n_clu * sizeof(*curr_cov));
                memcpy(eta, curr_eta, n_clu * sizeof(*curr_eta));
            }
        }
        free(curr_mean);
        free(curr_cov);
        free(curr_eta);
    }
    return 0;
}


int FitMSE(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* mag, const long n_clu, const long n_tries
) {
    /* Minimise the mean square error between a 2d gaussian mixture model and an image. */
    if (
        FitInit(
            roi, anchor_i, anchor_j, height, width,
            mean, cov, mag, n_clu
        )
        ||
        FitMSEMultiSteps(
            roi, anchor_i, anchor_j, height, width,
            mean, cov, mag, n_clu
        )
    ) {
        return 1;
    }
    if (n_tries > 1) {
        npy_float32 best_mse, mse;
        if (
            MSECost(
                roi, anchor_i, anchor_j, height, width,
                mean, cov, mag, n_clu,
                &best_mse
            )
        ) {
            return 1;
        }
        npy_float32 *curr_mean, *curr_cov, *curr_mag;
        curr_mean = malloc(2 * n_clu * sizeof(*curr_mean));
        if (curr_mean == NULL) {
            fprintf(stderr, "failed to malloc\n");
            return 1;
        }
        curr_cov = malloc(4 * n_clu * sizeof(*curr_cov));
        if (curr_cov == NULL) {
            free(curr_mean);
            fprintf(stderr, "failed to malloc\n");
            return 1;
        }
        curr_mag = malloc(n_clu * sizeof(*curr_mag));
        if (curr_mag == NULL) {
            free(curr_mean);
            free(curr_cov);
            fprintf(stderr, "failed to malloc\n");
            return 1;
        }
        for (long i = 1; i < n_tries; ++i) {
            if (
                FitInit(
                    roi, anchor_i, anchor_j, height, width,
                    curr_mean, curr_cov, curr_mag, n_clu
                )
                ||
                FitMSEMultiSteps(
                    roi, anchor_i, anchor_j, height, width,
                    curr_mean, curr_cov, curr_mag, n_clu
                )
                ||
                MSECost(
                    roi, anchor_i, anchor_j, height, width,
                    curr_mean, curr_cov, curr_mag, n_clu,
                    &mse
                )
            ) {
                free(curr_mean);
                free(curr_cov);
                free(curr_mag);
                return 1;
            }
            if (mse < best_mse) {
                best_mse = mse;
                memcpy(mean, curr_mean, 2 * n_clu * sizeof(*curr_mean));
                memcpy(cov, curr_cov, 4 * n_clu * sizeof(*curr_cov));
                memcpy(mag, curr_mag, n_clu * sizeof(*curr_mag));
            }
        }
        free(curr_mean);
        free(curr_cov);
        free(curr_mag);
    }
    return 0;
}


int BatchedFit(
    PyByteArrayObject* data, PyArrayObject* bboxes,
    npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu, const long n_tries,
    int (*func)(
        const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
        npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu, const long n_tries
    )
) {
    /* Apply the function func on each roi. */
    npy_float32* rawdata;
    npy_intp datalen = (npy_intp)PyByteArray_Size((PyObject *)data), shift = 0, area;
    const npy_intp n = PyArray_DIM(bboxes, 0);

    if (datalen % sizeof(npy_float)) {
        fprintf(stderr, "data length is not a multiple of float32 length\n");
        return 1;
    }
    datalen /= sizeof(npy_float);
    rawdata = (npy_float32 *)PyByteArray_AsString((PyObject *)data);
    if (rawdata == NULL) {
        fprintf(stderr, "data is empty\n");
        return 1;
    }

    srand(time(NULL));  // init random seed

    for (npy_intp i = 0; i < n; ++i) {
        long anchor_i = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 0);
        long anchor_j = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 1);
        long height = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 2);
        long width = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 3);
        area = (npy_intp)(height * width);
        if (!area) {
            fprintf(stderr, "the bbox %ld has zero area\n", i);
            return 1;
        }
        if (shift + area > datalen) {
            fprintf(stderr, "the data length 4*%ld is too short to fill roi of index %ld\n", datalen, i);
            return 1;
        }
        if (
            func(
                rawdata + shift, anchor_i, anchor_j, height, width,
                mean + 2 * n_clu * i, cov + 4 * n_clu * i, eta + n_clu * i, n_clu, n_tries
            )
        ) {
            fprintf(stderr, "the error comes for the roi %ld\n", i);
            return 1;
        }
        shift += area;
    }
    if (datalen != shift) {  // data to long
        fprintf(stderr, "the data length 4*%ld is too long to fill a total area of %ld pxls.\n", datalen, shift);
        return 1;
    }
    return 0;
}


// int BatchedFitParallel(
//     PyByteArrayObject* data, PyArrayObject* bboxes,
//     npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu, const long n_tries,
//     int (*func)(
//         const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
//         npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu, const long n_tries
//     )
// ) {
//     /* Apply the function func on each roi. */
//     npy_float32* rawdata;
//     npy_intp datalen = (npy_intp)PyByteArray_Size((PyObject *)data), area;
//     npy_intp* shift;
//     int error = 0;
//     const npy_intp n = PyArray_DIM(bboxes, 0);

//     if (datalen % sizeof(npy_float)) {
//         fprintf(stderr, "data length is not a multiple of float32 length\n");
//         return 1;
//     }
//     datalen /= sizeof(npy_float);
//     rawdata = (npy_float32 *)PyByteArray_AsString((PyObject *)data);
//     if (rawdata == NULL) {
//         fprintf(stderr, "data is empty\n");
//         return 1;
//     }

//     // precompute shift for parralel for
//     shift = malloc((n + 1) * sizeof(*shift));
//     if (shift == NULL) {
//         fprintf(stderr, "failed to malloc\n");
//         return 1;
//     }
//     shift[0] = 0;
//     for (npy_intp i = 0; i < n; ++i) {
//         long height = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 2);
//         long width = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 3);
//         area = (npy_intp)(height * width);
//         if (!area) {
//             fprintf(stderr, "the bbox %ld has zero area\n", i);
//             free(shift);
//             return 1;
//         }
//         if (shift[i] + area > datalen) {
//             fprintf(stderr, "the data length 4*%ld is too short to fill roi of index %ld\n", datalen, i);
//             free(shift);
//             return 1;
//         }
//         shift[i+1] = shift[i] + area;
//     }
//     if (datalen != shift[n]) {  // data to long
//         fprintf(stderr, "the data length 4*%ld is too long to fill a total area of %ld pxls.\n", datalen, shift[n]);
//         return 1;
//     }

//     srand(time(NULL));  // init random seed

//     // call function
//     #pragma omp parallel for schedule(static)
//     for (npy_intp i = 0; i < n; ++i) {
//         long anchor_i = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 0);
//         long anchor_j = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 1);
//         long height = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 2);
//         long width = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 3);
//         int local_error = func(
//             rawdata + shift[i], anchor_i, anchor_j, height, width,
//             mean + 2 * n_clu * i, cov + 4 * n_clu * i, eta + n_clu * i, n_clu, n_tries
//         );
//         #pragma omp atomic
//         error |= local_error;
//     }
//     free(shift);
//     return error;
// }


static PyObject* FitEmParser(PyObject* self, PyObject* args, PyObject* kwargs) {
    // Compute the grad of the mse loss between the predicted gmm and the rois
    static char *kwlist[] = {"data", "bboxes", "nbr_clusters", "nbr_tries", NULL};
    int error;
    long nbr_clu, nbr_tries;
    npy_intp shape[4];
    PyArrayObject *bboxes, *mean, *cov, *eta;
    PyByteArrayObject* data;
    PyObject* out;

    // parse and check
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "YO!ll", kwlist,
        &data, &PyArray_Type, &bboxes, &nbr_clu, &nbr_tries)
    ) {
        return NULL;
    }
    if (CheckBBoxes(bboxes)) {
        return NULL;
    }
    if (nbr_clu < 1 || nbr_tries < 1) {
        PyErr_SetString(PyExc_ValueError, "'nbr_clusters' and 'nbr_tries' must be >= 1");
        return NULL;
    }

    // alloc result
    shape[0] = PyArray_DIM(bboxes, 0);
    shape[1] = (npy_intp)nbr_clu;
    shape[2] = 2;
    shape[3] = 2;
    mean = (PyArrayObject *)PyArray_EMPTY(3, shape, NPY_FLOAT32, 0);  // c contiguous
    if (mean == NULL) {
        return PyErr_NoMemory();
    }
    cov = (PyArrayObject *)PyArray_EMPTY(4, shape, NPY_FLOAT32, 0);  // c contiguous
    if (cov == NULL) {
        Py_DECREF(mean);
        return PyErr_NoMemory();
    }
    eta = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
    if (eta == NULL) {
        Py_DECREF(mean);
        Py_DECREF(cov);
        return PyErr_NoMemory();
    }

    // fit em
    Py_BEGIN_ALLOW_THREADS
    error = BatchedFit(
        data, bboxes,
        (npy_float32 *)PyArray_DATA(mean), (npy_float32 *)PyArray_DATA(cov), (npy_float32 *)PyArray_DATA(eta), nbr_clu, nbr_tries,
        &FitEM
    );
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(mean);
        Py_DECREF(cov);
        Py_DECREF(eta);
        PyErr_SetString(PyExc_RuntimeError, "failed to fit em on each roi");
        return NULL;
    }

    // pack result
    out = PyTuple_Pack(3, mean, cov, eta);  // it doesn't steal reference
    Py_DECREF(mean);
    Py_DECREF(cov);
    Py_DECREF(eta);
    return out;
}


static PyObject* FitMSEParser(PyObject* self, PyObject* args, PyObject* kwargs) {
    // Compute the grad of the mse loss between the predicted gmm and the rois
    static char *kwlist[] = {"data", "bboxes", "nbr_clusters", "nbr_tries", NULL};
    int error;
    long nbr_clu, nbr_tries;
    npy_intp shape[4];
    PyArrayObject *bboxes, *mean, *cov, *mag;
    PyByteArrayObject* data;
    PyObject* out;

    // parse and check
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "YO!ll", kwlist,
        &data, &PyArray_Type, &bboxes, &nbr_clu, &nbr_tries)
    ) {
        return NULL;
    }
    if (CheckBBoxes(bboxes)) {
        return NULL;
    }
    if (nbr_clu < 1 || nbr_tries < 1) {
        PyErr_SetString(PyExc_ValueError, "'nbr_clusters' and 'nbr_tries' must be >= 1");
        return NULL;
    }

    // alloc result
    shape[0] = PyArray_DIM(bboxes, 0);
    shape[1] = (npy_intp)nbr_clu;
    shape[2] = 2;
    shape[3] = 2;
    mean = (PyArrayObject *)PyArray_EMPTY(3, shape, NPY_FLOAT32, 0);  // c contiguous
    if (mean == NULL) {
        return PyErr_NoMemory();
    }
    cov = (PyArrayObject *)PyArray_EMPTY(4, shape, NPY_FLOAT32, 0);  // c contiguous
    if (cov == NULL) {
        Py_DECREF(mean);
        return PyErr_NoMemory();
    }
    mag = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
    if (mag == NULL) {
        Py_DECREF(mean);
        Py_DECREF(cov);
        return PyErr_NoMemory();
    }

    // fit em
    Py_BEGIN_ALLOW_THREADS
    error = BatchedFit(
        data, bboxes,
        (npy_float32 *)PyArray_DATA(mean), (npy_float32 *)PyArray_DATA(cov), (npy_float32 *)PyArray_DATA(mag), nbr_clu, nbr_tries,
        &FitMSE
    );
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(mean);
        Py_DECREF(cov);
        Py_DECREF(mag);
        PyErr_SetString(PyExc_RuntimeError, "failed to fit em on each roi");
        return NULL;
    }

    // pack result
    out = PyTuple_Pack(3, mean, cov, mag);  // it doesn't steal reference
    Py_DECREF(mean);
    Py_DECREF(cov);
    Py_DECREF(mag);
    return out;
}


static PyMethodDef fitMethods[] = {
    {"fit_em", (PyCFunction)FitEmParser,  METH_VARARGS | METH_KEYWORDS, "Weighted version of the 2d EM algorithm."},
    {"fit_mse", (PyCFunction)FitMSEParser,  METH_VARARGS | METH_KEYWORDS, "Minimise the mse of 2d gmm."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_fit = {
    PyModuleDef_HEAD_INIT,
    "fit",
    "Fit 2d gmm.",
    -1,
    fitMethods
};


PyMODINIT_FUNC PyInit_c_fit(void) {
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&c_fit);
}
