/* Compact representation of rois tensor. */

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


int checkIndexs(PyArrayObject *indexes) {
    // Raise an exception if shape format is not correct.
    if (PyArray_NDIM(indexes) != 1) {
        PyErr_SetString(PyExc_ValueError, "'indexes' requires 1 dimensions");
        return 1;
    }
    if (PyArray_TYPE(indexes) != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError, "'indexes' has to be of type int64");
        return 1;
    }
    return 0;
}


int checkImg(PyArrayObject *img) {
    // Raise an exception if shape format is not correct.
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


int checkRois(PyArrayObject *rois) {
    // Raise an exception if rois format is not correct.
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


int checkShapes(PyArrayObject *shapes) {
    // Raise an exception if shape format is not correct.
    if (PyArray_NDIM(shapes) != 2) {
        PyErr_SetString(PyExc_ValueError, "'shapes' requires 2 dimensions");
        return 1;
    }
    if (PyArray_DIM(shapes, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "second axis of 'shapes' has to be of size 2, for height and width");
        return 1;
    }
    if (PyArray_TYPE(shapes) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "'shapes' has to be of type int32");
        return 1;
    }
    return 0;
}


int getDataIndexs(npy_int64 **data_indexes_p, PyArrayObject *bboxes) {
    // find the boundaries of unfoleded rois in data
    const npy_intp len_bboxes = PyArray_DIM(bboxes, 0);
    npy_intp i;
    npy_int32 height, width;
    *data_indexes_p = malloc((len_bboxes+1)*sizeof(npy_int64));
    if (NULL == data_indexes_p) {
        fprintf(stderr, "failed to alloc the cummulated bboxes area array\n");
        return 1;
    }
    **data_indexes_p = 0;
    for (i = 0; i < len_bboxes; ++i) {
        height = *(npy_int32 *)PyArray_GETPTR2(bboxes, i, 2);
        width = *(npy_int32 *)PyArray_GETPTR2(bboxes, i, 3);
        if (height <= 0 || width <= 0) {
            fprintf(stderr, "the area of the roi of index %ld if not > 0\n", i);
            return 1;
        }
        *(*data_indexes_p + i + 1) = *(*data_indexes_p + i) + (npy_int64)(height*width);
    }
    return 0;
}


int getHeightWidth(npy_intp *height, npy_intp *width, PyArrayObject *shapes) {
    // get the max height and max width for each bboxes
    const npy_intp n = PyArray_DIM(shapes, 0);
    npy_intp i;
    npy_int32 size;
    if (n == 0) {
        *height = 1, *width = 1;
        return 0;
    }
    *height = *(npy_int32 *)PyArray_GETPTR2(shapes, 0, 0);
    *width = *(npy_int32 *)PyArray_GETPTR2(shapes, 0, 1);
    for (i = 1; i < n; ++i) {
        size = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 0);
        if (size > *height) {
            *height = size;
        } else if (size <= 0) {
            fprintf(stderr, "the height %d roi of index %ld if not >= 1\n", size, i);
            return 1;
        }
        size = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 1);
        if (size > *width) {
            *width = size;
        } else if (size <= 0) {
            fprintf(stderr, "the width %d roi of index %ld if not >= 1\n", size, i);
            return 1;
        }
    }
    return 0;
}


int getDataLen(Py_ssize_t *datalen, PyArrayObject *shapes) {
    // Get the exact len neaded.
    *datalen = 0;
    npy_intp i;
    const npy_intp n = PyArray_DIM(shapes, 0);
    npy_int32 area;
    for (i = 0; i < n; ++i) {
        area = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 0) * *(npy_int32 *)PyArray_GETPTR2(shapes, i, 1);
        if (!area) {
            fprintf(stderr, "the bbox %ld has zero area\n", i);
            return 1;
        }
        *datalen += (Py_ssize_t)area;
    }
    return 0;
}


int getDataLenBboxes(Py_ssize_t *datalen, PyArrayObject *bboxes) {
    // Get the exact len neaded.
    *datalen = 0;
    npy_intp i;
    const npy_intp n = PyArray_DIM(bboxes, 0);
    npy_int32 area;
    for (i = 0; i < n; ++i) {
        area = *(npy_int32 *)PyArray_GETPTR2(bboxes, i, 2) * *(npy_int32 *)PyArray_GETPTR2(bboxes, i, 3);
        if (!area) {
            fprintf(stderr, "the bbox %ld has zero area\n", i);
            return 1;
        }
        *datalen += (Py_ssize_t)area;
    }
    return 0;
}


int fillRawfromBBoxesImg(npy_float *rawdata, PyArrayObject *bboxes, PyArrayObject *img_cont) {
    // copy the patches into the flatten rois.
    npy_intp i;
    const npy_intp n = PyArray_DIM(bboxes, 0);
    const npy_intp imgShape[2] = {PyArray_DIM(img_cont, 0), PyArray_DIM(img_cont, 1)};
    npy_int32 h, anchor_h, anchor_w, height, width;
    Py_ssize_t shift = 0;
    npy_float *imgdata;

    imgdata = PyArray_DATA(img_cont);

    for (i = 0; i < n; ++i) {
        anchor_h = *(npy_int32 *)PyArray_GETPTR2(bboxes, i, 0);
        anchor_w = *(npy_int32 *)PyArray_GETPTR2(bboxes, i, 1);
        height = *(npy_int32 *)PyArray_GETPTR2(bboxes, i, 2);
        width = *(npy_int32 *)PyArray_GETPTR2(bboxes, i, 3);
        if (anchor_h < 0 || anchor_w < 0 || anchor_h + height > imgShape[0] || anchor_w + width > imgShape[1]) {
            fprintf(stderr, "the roi of index %ld comes out of the picture\n", i);
            return 1;
        }
        for (h = anchor_h; h < anchor_h+height; ++h) {
            memcpy(rawdata + shift, imgdata + h*imgShape[1] + anchor_w, width*sizeof(npy_float));
            shift += width;
        }
    }
    return 0;
}


int fillRawfromShapesRois(npy_float *rawdata, PyArrayObject *shapes, PyArrayObject *rois) {
    // copy the rois into the flatten rois.
    npy_intp i;
    const npy_intp n = PyArray_DIM(shapes, 0);
    npy_int32 h, w, height, width;
    Py_ssize_t shift = 0;
    for (i = 0; i < n; ++i) {
        height = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 0);
        width = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 1);
        for (h = 0; h < height; ++h) {
            for (w = 0; w < width; ++w) {
                *(rawdata+shift) = *(npy_float *)PyArray_GETPTR3(rois, i, h, w);
                ++shift;
            }
        }
    }
    return 0;
}


int fillRois(PyArrayObject *rois, npy_float *data, const Py_ssize_t datalen, PyArrayObject *shapes) {
    // fill and pad the rois
    npy_intp i, stride;
    const npy_intp roisShape[3] = {PyArray_DIM(rois, 0), PyArray_DIM(rois, 1), PyArray_DIM(rois, 2)};
    npy_int32 h, height, width;
    Py_ssize_t shift = 0;
    npy_float *roisdata = PyArray_DATA(rois);  // ok because it is c contiguous
    for (i = 0; i < roisShape[0]; ++i) {
        height = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 0);
        width = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 1);
        if (shift + height*width > datalen) {
            fprintf(stderr, "the data length 4*%ld is too short to fill roi of index %ld\n", datalen, i);
            return 1;
        }
        stride = i*roisShape[1]*roisShape[2];
        for (h = 0; h < height; ++h) {
            memcpy(roisdata + stride + h*roisShape[2], data+shift, width*sizeof(npy_float));
            shift += width;
            memset(roisdata + stride + h*roisShape[2] + width, 0, (roisShape[2]-width)*sizeof(npy_float));
        }
        memset(roisdata + stride + height*roisShape[2], 0, (roisShape[2]*(roisShape[1]-height)*sizeof(npy_float)));
    }
    if (datalen != shift) {  // data to long
        fprintf(stderr, "the data length 4*%ld is too long to fill a total area of %ld pxls.\n", datalen, shift);
        return 1;
    }
    return 0;
}


int reorderBBoxes(PyArrayObject *newbboxes, Py_ssize_t *newdatalen, PyArrayObject *indexes, PyArrayObject *bboxes) {
    // fill the newbboxes in the order provided by indexes
    // find the new datalen
    // set the negative index positive
    const npy_intp len_indexes = PyArray_DIM(indexes, 0);
    const npy_intp len_bboxes = PyArray_DIM(bboxes, 0);
    npy_intp i;
    npy_int64 index;
    npy_int32 height, width;
    npy_int32 *newbboxesdata = PyArray_DATA(newbboxes);  // ok because it is c contiguous

    *newdatalen = 0;
    for (i = 0; i < len_indexes; ++i) {
        // get new index, set positive
        index = *(npy_int64 *)PyArray_GETPTR1(indexes, i);
        if (index < 0) {
            index += len_bboxes;
            if (index < 0) {
                fprintf(stderr, "the new index %ld of index %ld it too negative\n", index-len_bboxes, i);
                return 1;
            }
            *(npy_int64 *)PyArray_GETPTR1(indexes, i) = index;  // set index positive
        }
        else if (index >= len_bboxes) {
            fprintf(stderr, "the new index %ld of index %ld it too big\n", index, i);
            return 1;
        }
        // copy bbox and update newdatalen
        height = *(npy_int32 *)PyArray_GETPTR2(bboxes, index, 2);
        width = *(npy_int32 *)PyArray_GETPTR2(bboxes, index, 3);
        if ((height <= 0) | (width <= 0)) {
            fprintf(stderr, "the area of the roi of index %ld if not > 0\n", index);
            return 1;
        }
        *(newbboxesdata + 4*i) = *(npy_int32 *)PyArray_GETPTR2(bboxes, index, 0);
        *(newbboxesdata + 4*i + 1) = *(npy_int32 *)PyArray_GETPTR2(bboxes, index, 1);
        *(newbboxesdata + 4*i + 2) = height;
        *(newbboxesdata + 4*i + 3) = width;
        *newdatalen += height*width;
    }
    return 0;
}


int reorderData(PyObject *newdata, PyObject *data, PyArrayObject *indexes, npy_int64 *data_indexes) {
    const npy_intp len_indexes = PyArray_DIM(indexes, 0);
    npy_float *rawnewdata, *rawdata;
    npy_intp i;
    npy_int64 index, shift_in, shift_out, size;

    rawdata = (npy_float *)PyByteArray_AsString(data);
    rawnewdata = (npy_float *)PyByteArray_AsString(newdata);
    if (rawnewdata == NULL) {
        fprintf(stderr, "no content in 'newdata'\n");
        return 1;
    }
    if (rawdata == NULL) {
        fprintf(stderr, "no content in 'data'\n");
        return 1;
    }
    shift_out = 0;
    for (i = 0; i < len_indexes; ++i) {
        index = *(npy_int64 *)PyArray_GETPTR1(indexes, i);
        shift_in = data_indexes[index];
        size = data_indexes[index+1] - shift_in;
        memcpy(rawnewdata + shift_out, rawdata + shift_in, size*sizeof(npy_float));
        shift_out += size;
    }
    return 0;
}


static PyObject *filterByIndexs(PyObject *self, PyObject *args) {
    PyObject *data, *newdata;
    PyArrayObject *indexes, *bboxes, *newbboxes;
    npy_intp shape[2];
    Py_ssize_t newdatalen;
    int error;
    npy_int64 *data_indexes = NULL;
    PyObject *out;


    if (!PyArg_ParseTuple(args, "O!YO!", &PyArray_Type, &indexes, &data, &PyArray_Type, &bboxes)) {
        return NULL;
    }

    // verifications
    if (checkIndexs(indexes)) {
        return NULL;
    }
    if (checkBboxes(bboxes)) {
        return NULL;
    }

    // creation of the new bbox array
    shape[0] = PyArray_DIM(indexes, 0);
    shape[1] = 4;
    newbboxes = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_INT32, 0);  // c contiguous
    if (NULL == newbboxes) {
        return PyErr_NoMemory();
    }

    // fill new bbox and compute, cum old bbox area for data indexing
    Py_BEGIN_ALLOW_THREADS
    error = getDataIndexs(&data_indexes, bboxes);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(newbboxes);
        return PyErr_NoMemory();
    }

    // reorder the bboxes
    Py_BEGIN_ALLOW_THREADS
    error = reorderBBoxes(newbboxes, &newdatalen, indexes, bboxes);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(newbboxes);
        free(data_indexes);
        PyErr_SetString(PyExc_IndexError, "invalid indexes values");
        return NULL;
    }

    // reorder new data
    newdata = PyByteArray_FromStringAndSize(NULL, newdatalen*sizeof(npy_float));
    if (NULL == newdata) {
        Py_DECREF(newbboxes);
        free(data_indexes);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    error = reorderData(newdata, data, indexes, data_indexes);
    free(data_indexes);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(newbboxes);
        Py_DECREF(newdata);
        PyErr_SetString(PyExc_RuntimeError, "failed to reorganise data");
        return NULL;
    }

    // pack result
    out = PyTuple_Pack(2, newdata, newbboxes);  // it doesn't steal reference
    Py_DECREF(newdata);
    Py_DECREF(newbboxes);
    return out;
}


static PyObject *imgbboxes2raw(PyObject *self, PyObject *args) {
    PyObject *data = NULL;
    npy_float *rawdata = NULL;
    PyArrayObject *img, *img_cont, *bboxes;//, *shapes;
    Py_ssize_t datalen;
    int error;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img, &PyArray_Type, &bboxes)) {
        return NULL;
    }

    // verifications
    if (checkImg(img)) {
        return NULL;
    }
    if (checkBboxes(bboxes)) {
        return NULL;
    }

    // find data len
    // PyObject *slice_all = PySlice_New(NULL, NULL, NULL);
    // PyObject *two = PyLong_FromLong(2);
    // PyObject *slice_end = PySlice_New(two, NULL, NULL);
    // PyObject *all_slices = PyTuple_Pack(2, slice_all, slice_end);
    // shapes = (PyArrayObject *)PyObject_GetItem((PyObject *)bboxes, all_slices);
    // Py_DECREF(all_slices);
    // Py_DECREF(slice_all);
    // Py_DECREF(slice_end);
    // Py_DECREF(two);
    // Py_BEGIN_ALLOW_THREADS
    // error = getDataLen(&datalen, shapes);
    // Py_END_ALLOW_THREADS
    // Py_DECREF(shapes);
    // if (error) {
    //     PyErr_SetString(PyExc_ValueError, "some bboxes have area of zero");
    //     return NULL;
    // }
    Py_BEGIN_ALLOW_THREADS
    error = getDataLenBboxes(&datalen, bboxes);
    Py_END_ALLOW_THREADS
    if (error) {
        // Py_DECREF(img);
        // Py_DECREF(bboxes);
        PyErr_SetString(PyExc_ValueError, "some bboxes have area of zero");
        return NULL;
    }

    // create bytearray
    data = PyByteArray_FromStringAndSize(NULL, datalen*sizeof(npy_float)); // add check for NULL
    if (data == NULL) {
        return NULL;
    }
    rawdata = (npy_float *)PyByteArray_AsString(data);
    if (rawdata == NULL) {
        Py_DECREF(data);
        return NULL;
    }

    // fill bytes
    img_cont = PyArray_GETCONTIGUOUS(img);
    Py_BEGIN_ALLOW_THREADS
    error = fillRawfromBBoxesImg(rawdata, bboxes, img_cont);
    Py_END_ALLOW_THREADS
    Py_DECREF(img_cont);
    if (error) {
        Py_DECREF(data);
        PyErr_SetString(PyExc_ValueError, "failed to copy rois into data");
        return NULL;
    }

    return (PyObject *)data;
}


static PyObject *rawshapes2rois(PyObject *self, PyObject *args) {
    // create a torch array of rois from bytearray and bboxes dims
    PyArrayObject *rois, *shapes;
    npy_intp shape[3];
    PyByteArrayObject *data;
    npy_float *rawdata;
    Py_ssize_t datalen;
    int error;

    if (!PyArg_ParseTuple(args, "YO!", &data, &PyArray_Type, &shapes)) {
        return NULL;
    }
    rawdata = (npy_float *)PyByteArray_AsString((PyObject *)data);

    // verifications
    if (PyByteArray_Size((PyObject *)data) % sizeof(npy_float)) {
        PyErr_SetString(PyExc_ValueError, "data length is not a multiple of float32 length");
        return NULL;
    }
    if (NULL == rawdata ){
        PyErr_SetString(PyExc_ValueError, "data is empty");
        return NULL;
    }
    if (checkShapes(shapes)) {
        return NULL;
    }

    // alloc rois
    Py_BEGIN_ALLOW_THREADS
    shape[0] = PyArray_DIM(shapes, 0);
    error = getHeightWidth(&shape[1], &shape[2], shapes);
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_ValueError, "some shapes have area of zero");
        return NULL;
    }
    rois = (PyArrayObject *)PyArray_EMPTY(3, shape, NPY_FLOAT32, 0);  // c contiguous
    if (NULL == rois) {
        PyErr_NoMemory();
    }

    // fill rois
    Py_BEGIN_ALLOW_THREADS
    datalen = PyByteArray_Size((PyObject *)data) / sizeof(npy_float);
    error = fillRois(rois, rawdata, datalen, shapes);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(rois);
        PyErr_SetString(PyExc_ValueError, "data length dosen't match rois area");
        return NULL;
    }

    return (PyObject *)rois;
}


static PyObject *roisshapes2raw(PyObject *self, PyObject *args) {
    PyObject *data = NULL;
    npy_float *rawdata = NULL;
    PyArrayObject *rois, *shapes;
    Py_ssize_t datalen;
    int error;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &rois, &PyArray_Type, &shapes)) {
        return NULL;
    }

    // verifications
    if (checkShapes(shapes)) {
        return NULL;
    }
    if (checkRois(rois)) {
        return NULL;
    }
    if (PyArray_DIM(shapes, 0) != PyArray_DIM(rois, 0)) {
        PyErr_SetString(PyExc_ValueError, "'shapes' and 'rois' have not the same length");
        return NULL;
    }

    // find data len
    Py_BEGIN_ALLOW_THREADS
    error = getDataLen(&datalen, shapes);
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_ValueError, "some bboxes have area of zero");
        return NULL;
    }

    // create bytearray
    data = PyByteArray_FromStringAndSize(NULL, datalen*sizeof(npy_float)); // add check for NULL
    if (data == NULL) {
        return NULL;
    }
    rawdata = (npy_float *)PyByteArray_AsString(data);
    if (rawdata == NULL) {
        Py_DECREF(data);
        return NULL;
    }

    // fill bytes
    Py_BEGIN_ALLOW_THREADS
    error = fillRawfromShapesRois(rawdata, shapes, rois);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(data);
        PyErr_SetString(PyExc_ValueError, "failed to copy rois into data");
        return NULL;
    }

    return (PyObject *)data;
}


static PyMethodDef roisMethods[] = {
    {"filter_by_indexes", filterByIndexs, METH_VARARGS, "Select the rois of the given indexes."},
    {"imgbboxes2raw", imgbboxes2raw, METH_VARARGS, "Extract the rois from the image."},
    {"rawshapes2rois", rawshapes2rois, METH_VARARGS, "Unfold and pad the flatten rois data into a tensor."},
    {"roisshapes2raw", roisshapes2raw, METH_VARARGS, "Compress the rois into a flatten no padded respresentation."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_rois = {
    PyModuleDef_HEAD_INIT,
    "rois",
    "Efficient rois memory management.",
    -1,
    roisMethods
};


PyMODINIT_FUNC PyInit_c_rois(void) {
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&c_rois);
}
