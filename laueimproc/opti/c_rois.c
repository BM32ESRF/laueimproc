/* Compact representation of rois tensor. */

#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int FindDataBoundaries(npy_int64* data_indices_p[], PyArrayObject* bboxes) {
    // find the boundaries of unfoleded rois in data
    const npy_intp len_bboxes = PyArray_DIM(bboxes, 0);
    npy_int16 height, width;
    *data_indices_p = malloc((len_bboxes+1) * sizeof(**data_indices_p));
    if (data_indices_p == NULL) {
        fprintf(stderr, "failed to alloc the cummulated bboxes area array\n");
        return 1;
    }
    (*data_indices_p)[0] = 0;
    for (npy_intp i = 0; i < len_bboxes; ++i) {
        height = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 2);
        width = *(npy_int16 *)PyArray_GETPTR2(bboxes, i, 3);
        if (height <= 0 || width <= 0) {
            fprintf(stderr, "the area of the roi of index %ld if not > 0\n", i);
            return 1;
        }
        (*data_indices_p)[i + 1] = (*data_indices_p)[i] + (npy_int64)height * (npy_int64)width;
    }
    return 0;
}


int FillRawfromBBoxesImg(npy_float32* rawdata, PyArrayObject* bboxes, PyArrayObject* img_cont) {
    // copy the patches into the flatten rois.
    const npy_intp n = PyArray_DIM(bboxes, 0);
    const npy_intp img_shape[2] = {PyArray_DIM(img_cont, 0), PyArray_DIM(img_cont, 1)};
    npy_intp h, anchor_h, anchor_w, height, width;
    npy_intp shift = 0;
    npy_float32* imgdata;

    imgdata = PyArray_DATA(img_cont);

    for (npy_intp i = 0; i < n; ++i) {
        anchor_h = (npy_intp)(*(npy_int16 *)PyArray_GETPTR2(bboxes, i, 0));
        anchor_w = (npy_intp)(*(npy_int16 *)PyArray_GETPTR2(bboxes, i, 1));
        height = (npy_intp)(*(npy_int16 *)PyArray_GETPTR2(bboxes, i, 2));
        width = (npy_intp)(*(npy_int16 *)PyArray_GETPTR2(bboxes, i, 3));
        if (anchor_h < 0 || anchor_w < 0 || anchor_h + height > img_shape[0] || anchor_w + width > img_shape[1]) {
            fprintf(stderr, "the roi of index %ld comes out of the picture\n", i);
            return 1;
        }
        for (h = anchor_h; h < anchor_h+height; ++h) {
            memcpy(rawdata + shift, imgdata + h*img_shape[1] + anchor_w, width*sizeof(npy_float));
            shift += width;
        }
    }
    return 0;
}


int FillRawfromShapesRois(npy_float32* rawdata, PyArrayObject* shapes, PyArrayObject* rois) {
    // copy the rois into the flatten rois.
    const npy_intp n = PyArray_DIM(shapes, 0);
    npy_int16 height, width;
    npy_intp shift = 0;
    for (npy_intp i = 0; i < n; ++i) {
        height = *(npy_int16 *)PyArray_GETPTR2(shapes, i, 0);
        width = *(npy_int16 *)PyArray_GETPTR2(shapes, i, 1);
        for (npy_int16 h = 0; h < height; ++h) {
            for (npy_int16 w = 0; w < width; ++w) {
                rawdata[shift] = *(npy_float32 *)PyArray_GETPTR3(rois, i, h, w);
                ++shift;
            }
        }
    }
    return 0;
}


int FillRois(PyArrayObject* rois, npy_float32* data, const Py_ssize_t datalen, PyArrayObject* shapes) {
    // fill and pad the rois
    npy_intp stride;
    const npy_intp rois_shape[3] = {PyArray_DIM(rois, 0), PyArray_DIM(rois, 1), PyArray_DIM(rois, 2)};
    npy_intp height, width;
    npy_intp shift = 0;
    npy_float32 *roisdata = PyArray_DATA(rois);  // ok because it is c contiguous
    for (npy_intp i = 0; i < rois_shape[0]; ++i) {
        height = (npy_intp)(*(npy_int16 *)PyArray_GETPTR2(shapes, i, 0));
        width = (npy_intp)(*(npy_int16 *)PyArray_GETPTR2(shapes, i, 1));
        if (shift + height*width > (npy_intp)datalen) {
            fprintf(stderr, "the data length 4*%ld is too short to fill roi of index %ld\n", datalen, i);
            return 1;
        }
        stride = i * rois_shape[1] * rois_shape[2];
        for (npy_intp h = 0; h < height; ++h) {
            memcpy(roisdata + stride + h*rois_shape[2], data+shift, width*sizeof(npy_float));
            shift += width;
            memset(roisdata + stride + h*rois_shape[2] + width, 0, (rois_shape[2]-width)*sizeof(npy_float));
        }
        memset(roisdata + stride + height*rois_shape[2], 0, (rois_shape[2]*(rois_shape[1]-height)*sizeof(npy_float)));
    }
    if ((npy_intp)datalen != shift) {  // data to long
        fprintf(stderr, "the data length 4*%ld is too long to fill a total area of %ld pxls.\n", datalen, shift);
        return 1;
    }
    return 0;
}


int GetHeightWidth(npy_intp* height, npy_intp* width, PyArrayObject* shapes) {
    // get the max height and max width for each bboxes
    const npy_intp n = PyArray_DIM(shapes, 0);
    npy_intp i;
    npy_int16 size;
    if (n == 0) {
        *height = 1, *width = 1;
        return 0;
    }
    *height = (npy_intp)(*(npy_int16 *)PyArray_GETPTR2(shapes, 0, 0));
    *width = (npy_intp)(*(npy_int16 *)PyArray_GETPTR2(shapes, 0, 1));
    for (i = 1; i < n; ++i) {
        size = *(npy_int16 *)PyArray_GETPTR2(shapes, i, 0);
        if (size > *height) {  // max
            *height = (npy_intp)size;
        } else if (size <= 0) {
            fprintf(stderr, "the height %d roi of index %ld if not >= 1\n", size, i);
            return 1;
        }
        size = *(npy_int16 *)PyArray_GETPTR2(shapes, i, 1);
        if (size > *width) {
            *width = (npy_intp)size;
        } else if (size <= 0) {
            fprintf(stderr, "the width %d roi of index %ld if not >= 1\n", size, i);
            return 1;
        }
    }
    return 0;
}


int GetDataLen(Py_ssize_t* datalen, PyArrayObject* shapes_bboxes) {
    // Get the exact len needed.
    *datalen = 0;
    const npy_intp n = PyArray_DIM(shapes_bboxes, 0);
    const npy_intp shift = PyArray_DIM(shapes_bboxes, 1) - 2;
    unsigned long area;
    for (npy_intp i = 0; i < n; ++i) {
        area = (  // negative indexing to allow bboxes as well
            (unsigned long)(*(npy_int16 *)PyArray_GETPTR2(shapes_bboxes, i, shift))
            * (unsigned long)(*(npy_int16 *)PyArray_GETPTR2(shapes_bboxes, i, shift+1))
        );
        if (!area) {
            fprintf(stderr, "the bbox %ld has zero area\n", i);
            return 1;
        }
        *datalen += (Py_ssize_t)area;
    }
    return 0;
}


int ReorderBBoxes(PyArrayObject* newbboxes, Py_ssize_t* newdatalen, PyArrayObject* indices, PyArrayObject* bboxes) {
    // fill the newbboxes in the order provided by indices
    // find the new datalen
    // set the negative index positive
    const npy_intp len_indices = PyArray_DIM(indices, 0);
    const npy_intp len_bboxes = PyArray_DIM(bboxes, 0);
    npy_int64 index;
    npy_int16 height, width;
    npy_int16 *newbboxesdata = PyArray_DATA(newbboxes);  // ok because it is c contiguous

    *newdatalen = 0;
    for (npy_intp i = 0; i < len_indices; ++i) {
        // get new index, set positive
        index = *(npy_int64 *)PyArray_GETPTR1(indices, i);
        if (index < 0) {
            index += len_bboxes;
            if (index < 0) {
                fprintf(stderr, "the new index %ld of index %ld it too negative\n", index-len_bboxes, i);
                return 1;
            }
            *(npy_int64 *)PyArray_GETPTR1(indices, i) = index;  // set index positive
        } else if (index >= len_bboxes) {
            fprintf(stderr, "the new index %ld of index %ld it too big\n", index, i);
            return 1;
        }
        // copy bbox and update newdatalen
        height = *(npy_int16 *)PyArray_GETPTR2(bboxes, index, 2);
        width = *(npy_int16 *)PyArray_GETPTR2(bboxes, index, 3);
        if ((height <= 0) | (width <= 0)) {
            fprintf(stderr, "the area of the roi of index %ld if not > 0\n", index);
            return 1;
        }
        newbboxesdata[4*i] = *(npy_int16 *)PyArray_GETPTR2(bboxes, index, 0);
        newbboxesdata[4*i + 1] = *(npy_int16 *)PyArray_GETPTR2(bboxes, index, 1);
        newbboxesdata[4*i + 2] = height;
        newbboxesdata[4*i + 3] = width;
        *newdatalen += (Py_ssize_t)(height * width);
    }
    return 0;
}


int ReorderData(PyObject* newdata, PyObject* data, PyArrayObject* indices, npy_int64* data_indices) {
    const npy_intp len_indices = PyArray_DIM(indices, 0);
    npy_float32 *rawnewdata, *rawdata;
    npy_int64 index, shift_in, shift_out, size;

    rawdata = (npy_float32 *)PyByteArray_AsString(data);
    rawnewdata = (npy_float32 *)PyByteArray_AsString(newdata);
    if (rawnewdata == NULL) {
        fprintf(stderr, "no content in 'newdata'\n");
        return 1;
    }
    if (rawdata == NULL) {
        fprintf(stderr, "no content in 'data'\n");
        return 1;
    }
    shift_out = 0;
    for (npy_intp i = 0; i < len_indices; ++i) {
        index = *(npy_int64 *)PyArray_GETPTR1(indices, i);
        shift_in = data_indices[index];
        size = data_indices[index+1] - shift_in;
        memcpy(rawnewdata + shift_out, rawdata + shift_in, size*sizeof(npy_float));
        shift_out += size;
    }
    return 0;
}


static PyObject* FilterByIndices(PyObject* self, PyObject* args) {
    PyObject *data, *newdata;
    PyArrayObject *indices, *bboxes, *newbboxes;
    npy_intp shape[2];
    Py_ssize_t newdatalen;
    int error;
    npy_int64* data_indices = NULL;
    PyObject* out;


    if (!PyArg_ParseTuple(args, "O!YO!", &PyArray_Type, &indices, &data, &PyArray_Type, &bboxes)) {
        return NULL;
    }

    // verifications
    if (CheckIndices(indices)) {
        return NULL;
    }
    if (CheckBBoxes(bboxes)) {
        return NULL;
    }

    // creation of the new bbox array
    shape[0] = PyArray_DIM(indices, 0);
    shape[1] = 4;
    newbboxes = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_INT16, 0);  // c contiguous
    if (newbboxes == NULL) {
        return PyErr_NoMemory();
    }

    // fill new bbox and compute, cum old bbox area for data indexing
    Py_BEGIN_ALLOW_THREADS
    error = FindDataBoundaries(&data_indices, bboxes);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(newbboxes);
        return PyErr_NoMemory();
    }

    // reorder the bboxes
    Py_BEGIN_ALLOW_THREADS
    error = ReorderBBoxes(newbboxes, &newdatalen, indices, bboxes);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(newbboxes);
        free(data_indices);
        PyErr_SetString(PyExc_IndexError, "invalid indices values");
        return NULL;
    }

    // reorder new data
    newdata = PyByteArray_FromStringAndSize(NULL, newdatalen*sizeof(npy_float));
    if (newdata == NULL) {
        Py_DECREF(newbboxes);
        free(data_indices);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    error = ReorderData(newdata, data, indices, data_indices);
    free(data_indices);
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


static PyObject* ImgBBoxes2Raw(PyObject* self, PyObject* args) {
    PyObject* data = NULL;
    npy_float32* rawdata = NULL;
    PyArrayObject *img, *img_cont, *bboxes;
    Py_ssize_t datalen;
    int error;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img, &PyArray_Type, &bboxes)) {
        return NULL;
    }

    // verifications
    if (CheckImg(img, NPY_FLOAT32)) {
        return NULL;
    }
    if (CheckBBoxes(bboxes)) {
        return NULL;
    }

    // find data len
    Py_BEGIN_ALLOW_THREADS
    error = GetDataLen(&datalen, bboxes);
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_ValueError, "some bboxes have area of zero");
        return NULL;
    }

    // create bytearray
    data = PyByteArray_FromStringAndSize(NULL, datalen*sizeof(npy_float));
    if (data == NULL) {
        return NULL;
    }
    rawdata = (npy_float32 *)PyByteArray_AsString(data);
    if (rawdata == NULL) {
        Py_DECREF(data);
        return NULL;
    }

    // fill bytes
    img_cont = PyArray_GETCONTIGUOUS(img);
    Py_BEGIN_ALLOW_THREADS
    error = FillRawfromBBoxesImg(rawdata, bboxes, img_cont);
    Py_END_ALLOW_THREADS
    Py_DECREF(img_cont);
    if (error) {
        Py_DECREF(data);
        PyErr_SetString(PyExc_ValueError, "failed to copy rois into data");
        return NULL;
    }

    return (PyObject *)data;
}


static PyObject* RawShapes2Rois(PyObject* self, PyObject* args) {
    // create a torch array of rois from bytearray and bboxes dims
    PyArrayObject *rois, *shapes;
    npy_intp shape[3];
    PyByteArrayObject* data;
    npy_float32* rawdata;
    Py_ssize_t datalen;
    int error;

    if (!PyArg_ParseTuple(args, "YO!", &data, &PyArray_Type, &shapes)) {
        return NULL;
    }
    rawdata = (npy_float32 *)PyByteArray_AsString((PyObject *)data);

    // verifications
    if (PyByteArray_Size((PyObject *)data) % sizeof(npy_float)) {
        PyErr_SetString(PyExc_ValueError, "data length is not a multiple of float32 length");
        return NULL;
    }
    if (rawdata == NULL) {
        PyErr_SetString(PyExc_ValueError, "data is empty");
        return NULL;
    }
    if (CheckShapes(shapes)) {
        return NULL;
    }

    // alloc rois
    Py_BEGIN_ALLOW_THREADS
    shape[0] = PyArray_DIM(shapes, 0);
    error = GetHeightWidth(&shape[1], &shape[2], shapes);
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_ValueError, "some shapes have area of zero");
        return NULL;
    }
    rois = (PyArrayObject *)PyArray_EMPTY(3, shape, NPY_FLOAT32, 0);  // c contiguous
    if (rois == NULL) {
        return PyErr_NoMemory();
    }

    // fill rois
    Py_BEGIN_ALLOW_THREADS
    datalen = PyByteArray_Size((PyObject *)data) / sizeof(npy_float);
    error = FillRois(rois, rawdata, datalen, shapes);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(rois);
        PyErr_SetString(PyExc_ValueError, "data length dosen't match rois area");
        return NULL;
    }

    return (PyObject *)rois;
}


static PyObject* RoisShapes2Raw(PyObject* self, PyObject* args) {
    PyObject* data = NULL;
    npy_float32* rawdata = NULL;
    PyArrayObject *rois, *shapes;
    Py_ssize_t datalen;
    int error;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &rois, &PyArray_Type, &shapes)) {
        return NULL;
    }

    // verifications
    if (CheckShapes(shapes)) {
        return NULL;
    }
    if (CheckRois(rois)) {
        return NULL;
    }
    if (PyArray_DIM(shapes, 0) != PyArray_DIM(rois, 0)) {
        PyErr_SetString(PyExc_ValueError, "'shapes' and 'rois' have not the same length");
        return NULL;
    }

    // find data len
    Py_BEGIN_ALLOW_THREADS
    error = GetDataLen(&datalen, shapes);
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
    rawdata = (npy_float32 *)PyByteArray_AsString(data);
    if (rawdata == NULL) {
        Py_DECREF(data);
        return NULL;
    }

    // fill bytes
    Py_BEGIN_ALLOW_THREADS
    error = FillRawfromShapesRois(rawdata, shapes, rois);
    Py_END_ALLOW_THREADS
    if (error) {
        Py_DECREF(data);
        PyErr_SetString(PyExc_ValueError, "failed to copy rois into data");
        return NULL;
    }

    return (PyObject *)data;
}


static PyMethodDef roisMethods[] = {
    {"filter_by_indices", FilterByIndices, METH_VARARGS, "Select the rois of the given indices."},
    {"imgbboxes2raw", ImgBBoxes2Raw, METH_VARARGS, "Extract the rois from the image."},
    {"rawshapes2rois", RawShapes2Rois, METH_VARARGS, "Unfold and pad the flatten rois data into a tensor."},
    {"roisshapes2raw", RoisShapes2Raw, METH_VARARGS, "Compress the rois into a flatten no padded respresentation."},
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
