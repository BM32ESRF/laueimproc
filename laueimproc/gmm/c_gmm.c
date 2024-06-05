/* Multivariate gaussian computing. */

#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include "laueimproc/gmm/c_gmm.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int BatchedCost(
    PyByteArrayObject* data, PyArrayObject* bboxes,
    npy_float32* mean, npy_float32* cov, npy_float32* mag, const long n_clu,
    npy_float32* cost
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
            MSECost(
                rawdata + shift, anchor_i, anchor_j, height, width,
                mean + 2 * n_clu * i, cov + 4 * n_clu * i, mag + n_clu * i, n_clu,
                cost + i
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


int BatchedCostAndGrad(
    PyByteArrayObject* data, PyArrayObject* bboxes,
    npy_float32* mean, npy_float32* cov, npy_float32* mag, const long n_clu,
    npy_float32* cost, npy_float32* mean_grad, npy_float32* cov_grad, npy_float32* mag_grad
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
            MSECostAndGrad(
                rawdata + shift, anchor_i, anchor_j, height, width,
                mean + 2 * n_clu * i, cov + 4 * n_clu * i, mag + n_clu * i, n_clu,
                cost + i, mean_grad + 2 * n_clu * i, cov_grad + 4 * n_clu * i, mag_grad + n_clu * i
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


int CheckMeanCovMag(PyArrayObject* mean, PyArrayObject* cov, PyArrayObject* mag) {
    /* Check if the tensors have write shape, and dtype. */
    npy_intp n_batch, n_clu;
    if (PyArray_NDIM(mean) != 3) {
        PyErr_SetString(PyExc_ValueError, "'mean' requires 3 dimensions (n, k, 2)");
        return 1;;
    }
    if (PyArray_NDIM(cov) != 4) {
        PyErr_SetString(PyExc_ValueError, "'cov' requires 4 dimensions (n, k, 2, 2)");
        return 1;;
    }
    if (PyArray_NDIM(mag) != 2) {
        PyErr_SetString(PyExc_ValueError, "'mag' requires 2 dimensions (n, k)");
        return 1;;
    }
    if (PyArray_DIM(mean, 2) != 2) {
        PyErr_SetString(PyExc_ValueError, "fird axis of 'mean' has to be of size 2 (n, k, 2)");
        return 1;;
    }
    if (PyArray_DIM(cov, 2) != 2) {
        PyErr_SetString(PyExc_ValueError, "fird axis of 'cov' has to be of size 2 (n, k, 2, 2)");
        return 1;;
    }
    if (PyArray_DIM(cov, 3) != 2) {
        PyErr_SetString(PyExc_ValueError, "fourth axis of 'cov' has to be of size 2 (n, k, 2, 2)");
        return 1;;
    }
    n_batch = PyArray_DIM(mean, 0);
    if (PyArray_DIM(cov, 0) != n_batch) {
        PyErr_SetString(PyExc_ValueError, "batch dim of 'mean' and 'cov' has to be the same");
        return 1;;
    }
    if (PyArray_DIM(mag, 0) != n_batch) {
        PyErr_SetString(PyExc_ValueError, "batch dim of 'mean' and 'mag' has to be the same");
        return 1;;
    }
    n_clu = PyArray_DIM(mean, 1);
    if (PyArray_DIM(cov, 1) != n_clu) {
        PyErr_SetString(PyExc_ValueError, "cluster dim of 'mean' and 'cov' has to be the same");
        return 1;;
    }
    if (PyArray_DIM(mag, 1) != n_clu) {
        PyErr_SetString(PyExc_ValueError, "cluster dim of 'mean' and 'mag' has to be the same");
        return 1;;
    }
    if (PyArray_TYPE(mean) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "'mean' has to be of type float32");
        return 1;;
    }
    if (PyArray_TYPE(cov) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "'cov' has to be of type float32");
        return 1;;
    }
    if (PyArray_TYPE(mag) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "'mag' has to be of type float32");
        return 1;;
    }
    if (!PyArray_IS_C_CONTIGUOUS(mean)) {
        PyErr_SetString(PyExc_ValueError, "'mean' has to be c contiguous");
        return 1;;
    }
    if (!PyArray_IS_C_CONTIGUOUS(cov)) {
        PyErr_SetString(PyExc_ValueError, "'cov' has to be c contiguous");
        return 1;;
    }
    if (!PyArray_IS_C_CONTIGUOUS(mag)) {
        PyErr_SetString(PyExc_ValueError, "'mag' has to be c contiguous");
        return 1;;
    }
    return 0;
}


static PyObject* CostParser(PyObject* self, PyObject* args, PyObject* kwargs) {
    // Compute the mse loss between the predicted gmm and the rois
    static char *kwlist[] = {"data", "bboxes", "mean", "cov", "mag", NULL};
    int error;
    npy_intp shape[2];
    PyArrayObject *bboxes, *mean, *cov, *mag, *cost;
    PyByteArrayObject* data;

    // parse and check
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "YO!O!O!O!", kwlist,
        &data, &PyArray_Type, &bboxes, &PyArray_Type, &mean, &PyArray_Type, &cov, &PyArray_Type, &mag)
    ) {
        return NULL;
    }
    if (CheckBBoxes(bboxes)) {
        return NULL;
    }
    if (CheckMeanCovMag(mean, cov, mag)) {
        return NULL;
    }

    // alloc result
    shape[0] = PyArray_DIM(mean, 0);
    shape[1] = PyArray_DIM(mean, 1);
    cost = (PyArrayObject *)PyArray_EMPTY(1, shape, NPY_FLOAT32, 0);  // c contiguous
    if (cost == NULL) {
        return PyErr_NoMemory();
    }

    // compute cost and grad
    Py_BEGIN_ALLOW_THREADS
    error = BatchedCost(
        data, bboxes,
        (npy_float32 *)PyArray_DATA(mean), (npy_float32 *)PyArray_DATA(cov), (npy_float32 *)PyArray_DATA(mag), (long)shape[1],
        (npy_float32 *)PyArray_DATA(cost)
    );
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(cost);
        PyErr_SetString(PyExc_RuntimeError, "failed to compute cost on each roi");
        return NULL;
    }

    return (PyObject *)cost;
}


static PyObject* CostAndGradParser(PyObject* self, PyObject* args, PyObject* kwargs) {
    // Compute the grad of the mse loss between the predicted gmm and the rois
    static char *kwlist[] = {"data", "bboxes", "mean", "cov", "mag", NULL};
    PyArrayObject *bboxes, *mean, *cov, *mag, *cost, *mean_grad, *cov_grad, *mag_grad;
    PyByteArrayObject* data;
    PyObject* out;
    npy_intp shape[4];
    int error;

    // parse and check
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "YO!O!O!O!", kwlist,
        &data, &PyArray_Type, &bboxes, &PyArray_Type, &mean, &PyArray_Type, &cov, &PyArray_Type, &mag)
    ) {
        return NULL;
    }
    if (CheckBBoxes(bboxes)) {
        return NULL;
    }
    if (CheckMeanCovMag(mean, cov, mag)) {
        return NULL;
    }

    // alloc result
    shape[0] = PyArray_DIM(mean, 0);
    shape[1] = PyArray_DIM(mean, 1);
    shape[2] = 2;
    shape[3] = 2;
    cost = (PyArrayObject *)PyArray_EMPTY(1, shape, NPY_FLOAT32, 0);  // c contiguous
    if (cost == NULL) {
        return PyErr_NoMemory();
    }
    mean_grad = (PyArrayObject *)PyArray_EMPTY(3, shape, NPY_FLOAT32, 0);  // c contiguous
    if (mean_grad == NULL) {
        Py_DECREF(cost);
        return PyErr_NoMemory();
    }
    cov_grad = (PyArrayObject *)PyArray_EMPTY(4, shape, NPY_FLOAT32, 0);  // c contiguous
    if (cov_grad == NULL) {
        Py_DECREF(cost);
        Py_DECREF(mean_grad);
        return PyErr_NoMemory();
    }
    mag_grad = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
    if (mag_grad == NULL) {
        Py_DECREF(cost);
        Py_DECREF(mean_grad);
        Py_DECREF(cov_grad);
        return PyErr_NoMemory();
    }

    // compute cost and grad
    Py_BEGIN_ALLOW_THREADS
    error = BatchedCostAndGrad(
        data, bboxes,
        (npy_float32 *)PyArray_DATA(mean), (npy_float32 *)PyArray_DATA(cov), (npy_float32 *)PyArray_DATA(mag), (long)shape[1],
        (npy_float32 *)PyArray_DATA(cost), (npy_float32 *)PyArray_DATA(mean_grad), (npy_float32 *)PyArray_DATA(cov_grad), (npy_float32 *)PyArray_DATA(mag_grad)
    );
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(cost);
        Py_DECREF(mean_grad);
        Py_DECREF(cov_grad);
        Py_DECREF(mag_grad);
        PyErr_SetString(PyExc_RuntimeError, "failed to compute cost_and_grad on each roi");
        return NULL;
    }

    // pack result
    out = PyTuple_Pack(4, cost, mean_grad, cov_grad, mag_grad);  // it doesn't steal reference
    Py_DECREF(cost);
    Py_DECREF(mean_grad);
    Py_DECREF(cov_grad);
    Py_DECREF(mag_grad);
    return out;
}


static PyMethodDef gmmMethods[] = {
    {"mse_cost_and_grad", (PyCFunction)CostAndGradParser,  METH_VARARGS | METH_KEYWORDS, "Compute the grad of the mse between the predicted gmm and the rois."},
    {"mse_cost", (PyCFunction)CostParser,  METH_VARARGS | METH_KEYWORDS, "Compute the mse between the predicted gmm and the rois."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_gmm = {
    PyModuleDef_HEAD_INIT,
    "gmm",
    "Multivariate gaussian computing.",
    -1,
    gmmMethods
};


PyMODINIT_FUNC PyInit_c_gmm(void) {
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&c_gmm);
}
