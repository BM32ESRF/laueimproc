/* Performs basic checks on input python types and values. */

#define PY_SSIZE_T_CLEAN
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <Python.h>


int checkBboxes(PyArrayObject *bboxes) {
    // Raise an exception if shape format is not correct.
    if (PyArray_NDIM(bboxes) != 2) {
        PyErr_SetString(PyExc_ValueError, "'bboxes' requires 2 dimensions");
        return 1;
    }
    if (PyArray_DIM(bboxes, 1) != 4) {
        PyErr_SetString(
            PyExc_ValueError,
            "second axis of 'bboxes' has to be of size 4, for *anchors, height and width"
       );
        return 1;
    }
    if (PyArray_TYPE(bboxes) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "'bboxes' has to be of type int32");
        return 1;
    }
    return 0;
}
