/* Performs basic checks on input python types and values. */

#define PY_SSIZE_T_CLEAN
#include <numpy/arrayobject.h>
#include <Python.h>


int CheckBboxes(PyArrayObject* bboxes) {
    // Print an exception if shape or format is not correct.
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
    if (PyArray_TYPE(bboxes) != NPY_INT16) {
        PyErr_SetString(PyExc_TypeError, "'bboxes' has to be of type int16");
        return 1;
    }
    return 0;
}


int CheckImg(PyArrayObject* img) {
    // Print an exception if shape or format is not correct.
    if (PyArray_NDIM(img) != 2) {
        PyErr_SetString(PyExc_ValueError, "'img' requires 2 dimensions");
        return 1;
    }
    if (PyArray_TYPE(img) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "'img' has to be of type float32");
        return 1;
    }
    return 0;
}


int CheckIndices(PyArrayObject* indices) {
    // Print an exception if shape or format is not correct.
    if (PyArray_NDIM(indices) != 1) {
        PyErr_SetString(PyExc_ValueError, "'indices' requires 1 dimensions");
        return 1;
    }
    if (PyArray_TYPE(indices) != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError, "'indices' has to be of type int64");
        return 1;
    }
    return 0;
}


int CheckRois(PyArrayObject* rois) {
    // Print an exception if shape or format is not correct.
    if (PyArray_NDIM(rois) != 3) {
        PyErr_SetString(PyExc_ValueError, "'rois' requires 3 dimensions");
        return 1;
    }
    if (!PyArray_DIM(rois, 1)) {
        PyErr_SetString(PyExc_ValueError, "second axis of 'rois' has to be of non zero");
        return 1;
    }
    if (!PyArray_DIM(rois, 2)) {
        PyErr_SetString(PyExc_ValueError, "third axis of 'rois' has to be of non zero");
        return 1;
    }
    if (PyArray_TYPE(rois) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "'rois' has to be of type float32");
        return 1;
    }
    return 0;
}


int CheckShapes(PyArrayObject* shapes) {
    // Print an exception if shape or format is not correct.
    if (PyArray_NDIM(shapes) != 2) {
        PyErr_SetString(PyExc_ValueError, "'shapes' requires 2 dimensions");
        return 1;
    }
    if (PyArray_DIM(shapes, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "second axis of 'shapes' has to be of size 2, for height and width");
        return 1;
    }
    if (PyArray_TYPE(shapes) != NPY_INT16) {
        PyErr_SetString(PyExc_TypeError, "'shapes' has to be of type int16");
        return 1;
    }
    return 0;
}
