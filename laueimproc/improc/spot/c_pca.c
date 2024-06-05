/* Fast computation of PCA. */

#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include "laueimproc/gmm/c_linalg.h"
#include "laueimproc/improc/spot/c_spot_apply.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int RoiPCA(PyArrayObject* out, const npy_intp b, const npy_float32* roi, const npy_int16 bbox[4]) {
    // compute one PCA, store std1, std2 and theta in rawout
    npy_float32 mean[2], cov[3];
    if (ObsToMeanCov(roi, bbox, mean, cov)) {
        return 1;
    }
    if (Cov2dToEigtheta(&(cov[0]), &(cov[1]), &(cov[2]))) {
        return 1;
    }
    *(npy_float32 *)PyArray_GETPTR2(out, b, 0) = cov[0];  // std_1
    *(npy_float32 *)PyArray_GETPTR2(out, b, 1) = cov[1];  // std_2
    *(npy_float32 *)PyArray_GETPTR2(out, b, 2) = cov[2];  // theta
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
    if (CheckBBoxes(bboxes)) {
        return NULL;
    }

    shape[0] = PyArray_DIM(bboxes, 0);
    shape[1] = 3;
    pca = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
    if (pca == NULL) {
        return PyErr_NoMemory();
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
