/* Fast computation of PCA. */

#define PY_SSIZE_T_CLEAN
#include <laueimproc/c_check.h>
#include <laueimproc/improc/spot/c_spot_apply.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>

#define EPS 1.1920929e-7


int RoiPCA(PyArrayObject* out, const npy_intp i, const npy_float* roi, const npy_int16 bbox[4]) {
    // compute one PCA, store std1, std2 and theta in rawout
    long shift;
    npy_float weight, norm = 0, corr = 0;
    npy_float pos[2];
    npy_float mu[2] = {0, 0};
    npy_float sigma[2] = {0, 0};

    // get mean and total intensity
    for (long i = 0; i < bbox[2]; ++i) {
        shift = (long)bbox[3] * i;
        pos[0] = (npy_float)i;
        for (long j = 0; j < bbox[3]; ++j) {
            pos[1] = (npy_float)j;
            weight = roi[j + shift];
            mu[0] += pos[0] * weight, mu[1] += pos[1] * weight;  // SIMD
            norm += weight;
        }
    }
    if (!norm) {
        fprintf(stderr, "failed to compute pca because all the dup_w are equal to 0\n");
        return 1;
    }
    norm = 1 / norm;
    mu[0] *= norm; mu[1] *= norm;

    // get cov matrix
    for (long i = 0; i < bbox[2]; ++i) {
        shift = (long)bbox[3] * i;
        pos[0] = (npy_float)i - mu[0]; // i centered
        for (long j = 0; j < bbox[3]; ++j) {
            pos[1] = (npy_float)j - mu[1];  // j centered
            weight = roi[j + shift];
            weight *= weight;
            corr += weight * pos[0] * pos[1];
            sigma[0] += weight * pos[0] * pos[0], sigma[1] += weight * pos[1] * pos[1];  // SIMD
        }
    }
    norm *= norm;
    corr *= norm;
    sigma[0] *= norm, sigma[1] *= norm;

    // diagonalization
    corr *= 2;  // 2*c
    sigma[1] += sigma[0];  // s1+s2
    sigma[0] = sigma[1] - 2*sigma[0];  // s2-s1
    norm = corr*corr + sigma[0]*sigma[0];  // (2*c)**2 + (s2-s1)**2
    norm = sqrtf(norm);  // sqrt((2*c)**2 + (s2-s1)**2)
    if (fabsf(corr) > EPS) {
        sigma[0] += norm;  // s2 - s1 + sqrt((2*c)**2 + (s2-s1)**2)
        sigma[0] /= corr;  // (s2 - s1 + sqrt((2*c)**2 + (s2-s1)**2)) / (2*c)
        *(npy_float *)PyArray_GETPTR2(out, i, 2) = atanf(sigma[0]);  // theta
    }
    else {
        *(npy_float *)PyArray_GETPTR2(out, i, 2) = sigma[0] > EPS ? 0.5*M_PI : 0;  // theta
    }
    sigma[0] = sigma[1] - norm;  // s1 + s2 - sqrt((2*c)**2 + (s2-s1)**2)
    sigma[1] += norm;  // s1 + s2 + sqrt((2*c)**2 + (s2-s1)**2)
    sigma[0] *= 0.5;  // lambda2 = 1/2 * (s1 + s2 - sqrt((2*c)**2 + (s2-s1)**2))
    sigma[1] *= 0.5;  // lambda1 = 1/2 * (s1 + s2 + sqrt((2*c)**2 + (s2-s1)**2))
    *(npy_float *)PyArray_GETPTR2(out, i, 1) = sqrtf(sigma[0]);  // std2
    *(npy_float *)PyArray_GETPTR2(out, i, 0) = sqrtf(sigma[1]);  // std1
    return 0;
}


static PyObject* ComputeRoisPCA(PyObject* self, PyObject* args) {
    // Compute the PCA
    PyArrayObject *pca, *bboxes;
    PyByteArrayObject* data;
    npy_intp shape[2];
    int error;

    if (!PyArg_ParseTuple(args, "YO!", &data, &PyArray_Type, &bboxes)) {
        return NULL;
    }
    if (CheckBboxes(bboxes)) {
        return NULL;
    }

    shape[0] = PyArray_DIM(bboxes, 0);
    shape[1] = 3;
    pca = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
    if (pca == NULL) {
        PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    error = ApplyToRois(pca, data, bboxes, &RoiPCA);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(pca);
        PyErr_SetString(PyExc_RuntimeError, "failed to apply the roi_pca function on each roi");
        return NULL;
    }

    return (PyObject *)pca;
}


static PyMethodDef pcaMethods[] = {
    {"compute_rois_pca", ComputeRoisPCA, METH_VARARGS, "Compute the PCA of each roi."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_pca = {
    PyModuleDef_HEAD_INIT,
    "pca",
    "Fast Principal Componant Annalysis Calculus.",
    -1,
    pcaMethods
};


PyMODINIT_FUNC PyInit_c_pca(void) {
    import_array();
    if ( PyErr_Occurred() ) {
        return NULL;
    }
    return PyModule_Create(&c_pca);
}
