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
        cov[4 * j] = std1_std2_theta[0];
        cov[4 * j + 1] = std1_std2_theta[2];
        cov[4 * j + 2] = std1_std2_theta[2];
        cov[4 * j + 3] = std1_std2_theta[1];
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


int FitEM(
    const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
    npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu
) {
    return 0;
}


int BatchedFit(
    PyByteArrayObject* data, PyArrayObject* bboxes,
    npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu,
    int (*func)(
        const npy_float32* roi, const long anchor_i, const long anchor_j, const long height, const long width,
        npy_float32* mean, npy_float32* cov, npy_float32* eta, const long n_clu
    )
) {
    /* Apply the function CostAndGrad on each roi. */
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
            FitInit(
                rawdata + shift, anchor_i, anchor_j, height, width,
                mean + 2 * n_clu * i, cov + 4 * n_clu * i, eta + n_clu * i, n_clu
            )
            ||
            func(
                rawdata + shift, anchor_i, anchor_j, height, width,
                mean + 2 * n_clu * i, cov + 4 * n_clu * i, eta + n_clu * i, n_clu
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


static PyObject* FitEmParser(PyObject* self, PyObject* args, PyObject* kwargs) {
    // Compute the grad of the mse loss between the predicted gmm and the rois
    static char *kwlist[] = {"data", "bboxes", "nbr_clusters", "nbr_tries", NULL};
    int error;
    long nbr_clu = 1, nbr_tries = 2;
    npy_intp shape[4];
    PyArrayObject *bboxes, *mean, *cov, *eta;
    PyByteArrayObject* data;
    PyObject* out;

    // parse and check
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "YO!|ll", kwlist,
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
        (npy_float32 *)PyArray_DATA(mean), (npy_float32 *)PyArray_DATA(cov), (npy_float32 *)PyArray_DATA(eta), nbr_clu,
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


static PyMethodDef fitMethods[] = {
    {"fit_em", (PyCFunction)FitEmParser,  METH_VARARGS | METH_KEYWORDS, "Weighted version of the 2d EM algorithm."},
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
