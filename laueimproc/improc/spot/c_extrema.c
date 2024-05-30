/* Apply simple function on rois. */


#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include "laueimproc/improc/spot/c_spot_apply.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int RoiNbPeaks(PyArrayObject* out, const npy_intp i, const npy_float32* roi, const npy_int16 bbox[4]) {
    npy_int16 nbpeaks = 0;
    npy_float32 weight;
    long area = (long)bbox[2] * (long)bbox[3];
    long im1_s, i_s, ip1_s;
    long jm1, j, jp1;

    for (
        im1_s = -(long)bbox[3], i_s = 0, ip1_s = (long)bbox[3];
        i_s < area;
        im1_s = i_s, i_s = ip1_s, ip1_s += (long)bbox[3]
    ) {
        for (jm1 = -1, j = 0, jp1 = 1; j < (long)bbox[3]; jm1 = j, j = jp1++) {
            weight = roi[i_s + j];
            if (((i_s == 0 || j == 0) ? (npy_float)0 : roi[im1_s + jm1]) >= weight) continue;
            if (((i_s == 0) ? (npy_float)0 : roi[im1_s + j]) >= weight) continue;
            if (((i_s == 0 || jp1 == (long)bbox[3]) ? (npy_float)0 : roi[im1_s + jp1]) > weight) continue;
            if (((j == 0) ? (npy_float)0 : roi[i_s + jm1]) >= weight) continue;
            if (((jp1 == (long)bbox[3]) ? (npy_float)0 : roi[i_s + jp1]) > weight) continue;
            if (((ip1_s == area || j == 0) ? (npy_float)0 : roi[ip1_s + jm1]) >= weight) continue;
            if (((ip1_s == area) ? (npy_float)0 : roi[ip1_s + j]) > weight) continue;
            if (((ip1_s == area || jp1 == (long)bbox[3]) ? (npy_float)0 : roi[ip1_s + jp1]) > weight) continue;
            ++nbpeaks;
            jm1 = j, j = jp1++;  // optional, for optimization, skip the next loop
        }
    }

    *(npy_int16 *)PyArray_GETPTR1(out, i) = nbpeaks;
    return 0;
}


static PyObject* ComputeRoisNbPeaks(PyObject* self, PyObject* args) {
    // Compute the rois max
    PyArrayObject *nbpeaks, *bboxes;
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
    nbpeaks = (PyArrayObject *)PyArray_EMPTY(1, shape, NPY_INT16, 0);  // c contiguous
    if (nbpeaks == NULL) {
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    error = ApplyToRois(nbpeaks, data, bboxes, &RoiNbPeaks);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(nbpeaks);
        PyErr_SetString(PyExc_RuntimeError, "failed to apply the rois_nb_peaks function on each roi");
        return NULL;
    }

    return (PyObject *)nbpeaks;
}


static PyMethodDef extremaMethods[] = {
    {"compute_rois_nb_peaks", ComputeRoisNbPeaks, METH_VARARGS, "Compute the number of extremums of each roi."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_extrema = {
    PyModuleDef_HEAD_INIT,
    "basic",
    "Simple independant functions on each roi.",
    -1,
    extremaMethods
};


PyMODINIT_FUNC PyInit_c_extrema(void) {
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&c_extrema);
}
