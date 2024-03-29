/* Compact representation of rois tensor. */

#define PY_SSIZE_T_CLEAN
#include <numpy/arrayobject.h>
#include <Python.h>


int getHeightWidth(npy_intp *height, npy_intp *width, PyArrayObject *shapes) {
    // get the max height and max width for each bboxes
    const npy_intp n = PyArray_DIM(shapes, 0);
    npy_intp i;
    npy_int32 size;
    if ( n == 0 ) {
        *height = 1, *width = 1;
        return 0;
    }
    *height = *(npy_int32 *)PyArray_GETPTR2(shapes, 0, 0);
    *width = *(npy_int32 *)PyArray_GETPTR2(shapes, 0, 1);
    for ( i = 1; i < n; ++i ) {
        size = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 0);
        if ( size > *height ) {
            *height = size;
        }
        else if ( size <= 0 ) {
            PyErr_SetString(PyExc_ValueError, "some heights are not >= 1 (bbox with zero area)");
            return 1;
        }
        size = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 1);
        if ( size > *width ) {
            *width = size;
        }
        else if ( size <= 0 ) {
            PyErr_SetString(PyExc_ValueError, "some widths are not >= 1 (bbox with zero area)");
            return 1;
        }
    }
    return 0;
}

int fillRois(PyArrayObject *rois, const npy_float *data, const Py_ssize_t datalen, PyArrayObject *shapes) {
    // fill and pad the rois
    const npy_intp *roisShape = PyArray_SHAPE(shapes);
    npy_intp i;
    npy_int32 h, w, height, width;
    Py_ssize_t shift = 0;
    for ( i = 0; i < roisShape[0]; ++i ) {
        height = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 0);
        width = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 1);
        if ( shift + height*width > datalen ) {
            PyErr_SetString(PyExc_ValueError, "flatten rois buffer is not long enough");
            return 1;
        }
        for ( h = 0; h < height; ++h ) {
            for ( w = 0; w < width; ++w ) {
                *(npy_float *)PyArray_GETPTR3(rois, i, h, w) = *(data+shift+w);
            }
            shift += width;
            for ( w = width; w < roisShape[1]; ++w ) {
                *(npy_float *)PyArray_GETPTR3(rois, i, h, w) = 0;
            }
        }
        for ( h = height; h < roisShape[2]; ++h ) {
            for ( w = 0; w < roisShape[1]; ++w ) {
                *(npy_float *)PyArray_GETPTR3(rois, i, h, w) = 0;
            }
        }
    }
    return 0;
}


static PyObject *bin2rois(PyObject *self, PyObject *args) {
    // create a torch array of rois from bytearray and bboxes dims
    PyByteArrayObject *data;
    const npy_float *rawdata;
    Py_ssize_t datalen;
    PyArrayObject *shapes, *rois;
    npy_intp shape[3];
    int error;

    if ( !PyArg_ParseTuple(args, "YO!", &data, &PyArray_Type, &shapes) ) {
        return NULL;
    }

    // verifications
    if ( PyArray_NDIM(shapes) != 2 ) {
        PyErr_SetString(PyExc_ValueError, "'shapes' requires 2 dimensions");
        return NULL;
    }
    if ( PyArray_DIM(shapes, 1) != 2 ) {
        PyErr_SetString(PyExc_ValueError, "second axis of 'shapes' has to be of size 2, for height and width");
        return NULL;
    }
    if ( PyArray_TYPE(shapes) != NPY_INT32 ) {
        PyErr_SetString(PyExc_TypeError, "'shapes' has to be of type int32");
        return NULL;
    }

    // alloc rois
    Py_BEGIN_ALLOW_THREADS
    shape[0] = PyArray_DIM(shapes, 0);
    error = getHeightWidth(&shape[1], &shape[2], shapes);
    Py_END_ALLOW_THREADS
    if ( error ) {
        return NULL;
    }
    rois = (PyArrayObject *)PyArray_SimpleNew(3, shape, NPY_FLOAT32);
    if ( NULL == rois ) {
        PyErr_SetString(PyExc_RuntimeError, "failed to alloc the new array 'rois'");
        return NULL;
    }

    // fill rois
    Py_BEGIN_ALLOW_THREADS
    rawdata = (npy_float *)PyByteArray_AsString((PyObject *)data);
    datalen = PyByteArray_Size((PyObject *)data) / sizeof(npy_float);
    error = fillRois(rois, rawdata, datalen, shapes);
    Py_END_ALLOW_THREADS
    if ( error ) {
        return NULL;
    }

    return (PyObject *)rois;
}


static PyMethodDef bin2roisMethods[] = {
    {"bin2rois", bin2rois, METH_VARARGS, "Unfold and pad the flatten rois data into a tensor."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef c_rois = {
    PyModuleDef_HEAD_INIT,
    "rois",
    "Efficient rois memory management.",
    -1,
    bin2roisMethods
};

PyMODINIT_FUNC PyInit_c_rois(void) {
    import_array();
    return PyModule_Create(&c_rois);
}
