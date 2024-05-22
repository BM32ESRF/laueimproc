/* Apply simple function on rois. */


#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include "laueimproc/improc/spot/c_spot_apply.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int RoiCentroid(PyArrayObject* out, const npy_intp i, const npy_float* roi, const npy_int16 bbox[4]) {
    npy_float weight, total_weight = 0;
    long shift;
    npy_float pos[2], bary[2] = {0, 0};
    for (long i = 0; i < bbox[2]; ++i) {
        shift = (long)bbox[3] * i;
        pos[0] = (npy_float)i;
        for (long j = 0; j < bbox[3]; ++j) {
            pos[1] = (npy_float)j;
            weight = roi[j + shift];
            bary[0] += pos[0] * weight, bary[1] += pos[1] * weight;  // SIMD
            total_weight += weight;
        }
    }
    total_weight = 1 / total_weight;
    bary[0] *= total_weight; bary[1] *= total_weight;
    bary[0] += 0.5; bary[1] += 0.5;
    bary[0] += (npy_float)bbox[0]; bary[1] += (npy_float)bbox[1];  // relative to absolute base
    *(npy_float *)PyArray_GETPTR2(out, i, 0) = bary[0];
    *(npy_float *)PyArray_GETPTR2(out, i, 1) = bary[1];
    return 0;
}


int RoiInertia(PyArrayObject* out, const npy_intp i, const npy_float* roi, const npy_int16 bbox[4]) {
    npy_float weight, inertia = 0;
    long shift;
    npy_float pos[2], bary[2] = {0, 0};

    // barycenter
    for (long i = 0; i < bbox[2]; ++i) {
        shift = (long)bbox[3] * i;
        pos[0] = (npy_float)i;
        for (long j = 0; j < bbox[3]; ++j) {
            pos[1] = (npy_float)j;
            weight = roi[j + shift];
            bary[0] += pos[0] * weight, bary[1] += pos[1] * weight;  // SIMD
            inertia += weight;
        }
    }
    inertia = 1 / inertia;  // inertia is the total weight here
    bary[0] *= inertia; bary[1] *= inertia;

    // inertia
    inertia = 0;
    for (long i = 0; i < bbox[2]; ++i) {
        shift = (long)bbox[3] * i;
        pos[0] = (npy_float)i - bary[0];
        pos[0] *= pos[0];  // ri**2
        for (long j = 0; j < bbox[3]; ++j) {
            pos[1] = (npy_float)j - bary[1];
            pos[1] *= pos[1];  // rj**2
            weight = roi[j + shift];
            inertia += weight * (pos[0] + pos[1]);  // m * r**2
        }
    }
    *(npy_float *)PyArray_GETPTR1(out, i) = inertia;
    return 0;
}


int RoiMax(PyArrayObject* out, const npy_intp i, const npy_float* roi, const npy_int16 bbox[4]) {
    npy_float max = roi[0];
    long argmax = 0;
    for (long k = 1; k < bbox[2]*bbox[3]; ++k) {
        if (roi[k] > max) {
            argmax = k;
            max = roi[k];
        }
    }
    *(npy_float *)PyArray_GETPTR2(out, i, 0) = (npy_float)bbox[0] + 0.5 + (npy_float)(argmax / bbox[3]);
    *(npy_float *)PyArray_GETPTR2(out, i, 1) = (npy_float)bbox[1] + 0.5 + (npy_float)(argmax % bbox[3]);
    *(npy_float *)PyArray_GETPTR2(out, i, 2) = max;
    return 0;
}


int RoiSum(PyArrayObject* out, const npy_intp i, const npy_float* roi, const npy_int16 bbox[4]) {
    npy_float sum = roi[0];
    for (long k = 1; k < bbox[2]*bbox[3]; ++k) {
        sum += roi[k];
    }
    *(npy_float *)PyArray_GETPTR1(out, i) = sum;
    return 0;
}


static PyObject* ComputeRoisCentroid(PyObject* self, PyObject* args) {
    // Compute the barycenters
    PyArrayObject *barycenters, *bboxes;
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
    shape[1] = 2;
    barycenters = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
    if (barycenters == NULL) {
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    error = ApplyToRois(barycenters, data, bboxes, &RoiCentroid);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(barycenters);
        PyErr_SetString(PyExc_RuntimeError, "failed to apply the barycenter function on each roi");
        return NULL;
    }

    return (PyObject *)barycenters;
}


static PyObject* ComputeRoisInertia(PyObject* self, PyObject* args) {
    // Compute the rois max
    PyArrayObject *roisinertia, *bboxes;
    PyByteArrayObject* data;
    npy_intp shape[1];
    int error;

    if (!PyArg_ParseTuple(args, "YO!", &data, &PyArray_Type, &bboxes)) {
        return NULL;
    }
    if (CheckBBoxes(bboxes)) {
        return NULL;
    }

    shape[0] = PyArray_DIM(bboxes, 0);
    roisinertia = (PyArrayObject *)PyArray_EMPTY(1, shape, NPY_FLOAT32, 0);  // c contiguous
    if (roisinertia == NULL) {
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    error = ApplyToRois(roisinertia, data, bboxes, &RoiInertia);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(roisinertia);
        PyErr_SetString(PyExc_RuntimeError, "failed to apply the rois_inertia function on each roi");
        return NULL;
    }

    return (PyObject *)roisinertia;
}


static PyObject* ComputeRoisMax(PyObject* self, PyObject* args) {
    // Compute the rois max
    PyArrayObject *roismax, *bboxes;
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
    roismax = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
    if (roismax == NULL) {
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    error = ApplyToRois(roismax, data, bboxes, &RoiMax);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(roismax);
        PyErr_SetString(PyExc_RuntimeError, "failed to apply the rois_max function on each roi");
        return NULL;
    }

    return (PyObject *)roismax;
}


static PyObject* ComputeRoisSum(PyObject* self, PyObject* args) {
    // Compute the rois max
    PyArrayObject *roissum, *bboxes;
    PyByteArrayObject* data;
    npy_intp shape[1];
    int error;

    if (!PyArg_ParseTuple(args, "YO!", &data, &PyArray_Type, &bboxes)) {
        return NULL;
    }
    if (CheckBBoxes(bboxes)) {
        return NULL;
    }

    shape[0] = PyArray_DIM(bboxes, 0);
    roissum = (PyArrayObject *)PyArray_EMPTY(1, shape, NPY_FLOAT32, 0);  // c contiguous
    if (roissum == NULL) {
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    error = ApplyToRois(roissum, data, bboxes, &RoiSum);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(roissum);
        PyErr_SetString(PyExc_RuntimeError, "failed to apply the rois_sum function on each roi");
        return NULL;
    }

    return (PyObject *)roissum;
}


static PyMethodDef basicMethods[] = {
    {"compute_rois_centroid", ComputeRoisCentroid, METH_VARARGS, "Compute the weighted barycenter of each roi."},
    {"compute_rois_inertia", ComputeRoisInertia, METH_VARARGS, "Compute the order 2 momentum centered along the barycenter."},
    {"compute_rois_max", ComputeRoisMax, METH_VARARGS, "Compute the argmax of the intensity and the intensity max of each roi."},
    {"compute_rois_sum", ComputeRoisSum, METH_VARARGS, "Compute the sum of the intensities of each roi."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_basic = {
    PyModuleDef_HEAD_INIT,
    "basic",
    "Simple independant functions on each roi.",
    -1,
    basicMethods
};


PyMODINIT_FUNC PyInit_c_basic(void) {
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&c_basic);
}
