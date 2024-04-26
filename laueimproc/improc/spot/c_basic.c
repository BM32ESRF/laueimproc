/* Apply simple function on rois. */


#define PY_SSIZE_T_CLEAN
#include <laueimproc/c_check.h>
#include <laueimproc/improc/spot/c_spot_apply.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int Barycenter(PyArrayObject* out, const npy_intp i, const npy_float* roi, const npy_int16 bbox[4]) {
    npy_float weight, total_weight = 0;
    long shift;
    npy_float pos[2];
    npy_float bary[2] = {0, 0};
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


static PyObject* ComputeBarycenters(PyObject* self, PyObject* args) {
    // Compute the barycenters
    PyArrayObject *barycenters, *bboxes;
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
    shape[1] = 2;
    barycenters = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
    if (barycenters == NULL) {
        PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    error = ApplyToRois(barycenters, data, bboxes, &Barycenter);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(barycenters);
        PyErr_SetString(PyExc_RuntimeError, "failed to apply the barycenter function on each roi");
        return NULL;
    }

    return (PyObject *)barycenters;
}


static PyMethodDef basicMethods[] = {
    {"compute_barycenters", ComputeBarycenters, METH_VARARGS, "Compute the weighted barycenter of each roi."},
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
