/* Morphological operations. */


#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <omp.h>  // for thread
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>  // for malloc


#pragma omp declare simd
npy_float32 max(const npy_float32 val1, const npy_float32 val2) {
    return val1 > val2 ? val1 : val2;
}


#pragma omp declare simd
npy_float32 min(const npy_float32 val1, const npy_float32 val2) {
    return val1 < val2 ? val1 : val2;
}


int GetKernel(const float radius, long* kernel[], long* ksize) {
    /* Find the radius of each line of the rond kernel. */
    *ksize = (long)ceilf(radius - 0.5);  // len(real kernel) = *ksize * 2 - 1
    const float radius_square = radius * radius;
    float i_square;

    // alloc
    *kernel = calloc(*ksize, sizeof(**kernel));
    if (*kernel == NULL) {
        fprintf(stderr, "failed to malloc kernel\n");
        return 1;
    }

    // fill
    for (long i = 1; i < *ksize + 1; ++i) {
        i_square = (float)(i * i);
        for (long j = *ksize; j > 0; --j) {
            if (i_square + (float)(j * j) <= radius_square) {
                (*kernel)[*ksize - i] = j;
                break;
            }
        }
    }

    // for (long i = 0; i < *ksize; ++i) {
    //     fprintf(stderr, "i=%ld, kernel=%ld\n", i, (*kernel)[i]);
    // }

    return 0;
}


int UpdateLine(
    npy_float32* src,  // the source lines
    npy_float32* dst,  // the destination lines
    const long src_len,  // the number of elements aglomerated in the source line
    const long items,  // the number of items, len of the vectors src and dst
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)  // atomic comparison function min or max
) {
    /*
        Compare two or three elements, store the result for the next step and update the image.
        'src_len' is assumed to be odd numbers.
        'src' and 'dst' can be the same pointer.
    */
    if (src_len == 1) {
        // src_len = 1 -> dst_len = 3
        dst[0] = comp(src[0], src[1]);
        #pragma omp simd safelen(2)
        for (long j = 2; j < items - 1; j += 2) {  // fill index j and j - 1 in each loop
            npy_float32 buff = comp(src[j - 1], src[j]);
            dst[j - 1] = comp(src[j - 2], buff);
            dst[j] = comp(buff, src[j + 1]);
        }
        if (src == dst) {
            dst[items - 1] = dst[items - 2];
        } else {
            dst[items - 1] = comp(src[items - 2], src[items - 1]);
        }
        return 0;
    }

    if (src == dst) {
        fprintf(stderr, "src has to be different from dst\n");
        return 1;
        // // src_len = 2n + 1 -> dst_len = 2n + 3, with n >= 1
        // npy_float32 prev = src[0];
        // dst[0] = src[1];
        // for (long j = 1; j < items - 1; ++j) {
        //     npy_float32 local = src[j];
        //     dst[j] = comp(prev, src[j + 1]);
        //     prev = local;
        // }
        // dst[items - 1] = prev;
    }

    // src_len = 2n + 1 -> dst_len = 2n + 3, with n >= 1
    dst[0] = src[1];
    #pragma omp simd safelen(4)
    for (long j = 1; j < items - 1; ++j) {
        dst[j] = comp(src[j - 1], src[j + 1]);
    }
    dst[items - 1] = src[items - 2];

    return 0;
}


int UpdateLineAllLayers(
    npy_float32* img,  // src image
    npy_float32* dst,  // data to share comparisons result
    const long nb_layers,  // minimum 1 if kernel is (1, 1), 2 if kernel (3, 3), ...
    const long height,  // height of image
    const long width,  // wigth of image
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)  // atomic comparison function min or max
) {
    /*
        Compare the elements on one lines on left and on right, store the result in next layer.
        Iter the processus until reaching the number of required layers.
        The last layer is stored in the source.
        With is assumed to be >= 3.
    */
    const long layer_stride = height * width;
    int error;

    // first layer
    memcpy(dst, img, width * sizeof(*img));

    // medium layers
    for (long k = 1; k < nb_layers; ++k) {
        error = UpdateLine(dst + (k - 1) * layer_stride, dst + k * layer_stride, 2 * k - 1, width, comp);
        if (error) {
            return error;
        }
    }

    // last layer, stored into img
    return UpdateLine(
        dst + (nb_layers - 1) * layer_stride,
        img,
        2 * nb_layers - 1,
        width,
        comp
    );
}


void CompFirstLine(
    npy_float32* line_src,
    npy_float32* line_dst,
    const long width,
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2),
    void (*callback)(
        npy_float32* line,
        const long width,
        const npy_float32 elem,
        const long j,
        npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)
    )
) {
    /*
        Group the elements of the line by 3, call a function by given the comparaison of the 3 elements.
    */
    npy_float32 elem;

    // left
    elem = comp(line_src[0], line_src[1]);
    callback(line_dst, width, elem, 0, comp);

    // middle
    for (long j = 1; j < width - 1; ++j) {
        elem = comp(comp(line_src[j - 1], line_src[j]), line_src[j + 1]);
        callback(line_dst, width, elem, j, comp);
    }

    // end
    elem = comp(line_src[width - 1], line_src[width - 2]);
    callback(line_dst, width, elem, width - 1, comp);
}


int Morpho_1(
    npy_float32* img, npy_float32* out, const long height, const long width,
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)
) {
    /*
        For kernel = [[1]]
    */
    memcpy(out, img, height * width * sizeof(*img));
    return 0;
}


void callback_1_3(
    npy_float32* line,
    const long width,
    const npy_float32 elem,
    const long j,
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)
) {
    fprintf(stderr, "toto %ld\n", line);
    line[j] = comp(comp(line[j - width], line[j + width]), elem);
}


int Morpho_1_3(
    npy_float32* img, npy_float32* out, const long height, const long width,
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)
) {
    /*
        For kernel = [[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]]
    */
    const long final_shift = (height - 1) * width;

    UpdateLine(img, out, 1, width, comp);
    if (height >= 2) {
        #pragma omp simd
        for (long j = 0; j < width; ++j) {
            out[j] = comp(out[j], img[j + width]);
        }
    }

    for (long line_shift = width; line_shift < final_shift; line_shift += width) {
        CompFirstLine(img + line_shift, out + line_shift, width, comp, &callback_1_3);
    }

    UpdateLine(img + final_shift, out + final_shift, 1, width, comp);
    if (height >= 3) {
        #pragma omp simd
        for (long j = 0; j < width; ++j) {
            out[j + final_shift] = comp(out[j + final_shift], img[j + final_shift - width]);
        }
    }

    return 0;
}


int Morphology(PyArrayObject* img, const float radius, npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)) {
    /* Apply the morphological simple operation (func comp) on the image img with a circular kernel */

    int error;
    const long height = (long)PyArray_DIM(img, 0), width = (long)PyArray_DIM(img, 1);
    const long layer_stride = height * width;
    npy_float32* img_data = (npy_float32* )PyArray_DATA(img);
    npy_float32* layers;
    long* kernel;
    long ksize;  // kernel half size

    // verifications
    if (width <= 4) {
        fprintf(stderr, "image too few, width has to be >= 4\n");
        return 1;
    }

    // retrive structurant element shape
    error = GetKernel(radius, &kernel, &ksize);
    if (error) {
        return error;
    }

    // allocation compare layers
    layers = malloc(height * width * ksize * sizeof(*layers));
    if (layers == NULL) {
        fprintf(stderr, "failed to malloc comp layers\n");
        return 1;
    }

    // fill compare layers
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < height; ++i) {
        UpdateLineAllLayers(img_data + i * width, layers + i * width, ksize, height, width, comp);
    }

    // compare with kernel
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < height; ++i) {
        // top
        long is = i * width;
        for (long l = i - ksize >= 0 ? 0 : ksize - i; l < ksize; ++l) {
            long shift = (i - ksize + l) * width + kernel[l] * layer_stride;
            #pragma omp simd safelen(4)
            for (long j = 0; j < width; ++j) {
                img_data[j + is] = comp(img_data[j + is], layers[j + shift]);  // layers[kernel[l]][i-ksize+l][j]
            }
        }

        // bottom
        for (long l = i + ksize < height ? 0 : ksize + i + 1 - height; l < ksize; ++l) {
            long shift = (i + ksize - l) * width + kernel[l] * layer_stride;
            #pragma omp simd safelen(4)
            for (long j = 0; j < width; ++j) {
                img_data[j + is] = comp(img_data[j + is], layers[j + shift]);  // layers[kernel[l]][i+ksize-l][j]
            }
        }
    }

    free(layers);

    return 0;
}


int CClose(PyArrayObject* img, const float radius) {
    int error = Morphology(img, radius, &max);
    if (error) {
        return error;
    }
    return Morphology(img, radius, &min);
}


int COpen(PyArrayObject* img, const float radius) {
    int error = Morphology(img, radius, &min);
    if (error) {
        return error;
    }
    return Morphology(img, radius, &max);
}


static PyObject* Close(PyObject* self, PyObject* args) {
    // Morphological closing
    PyArrayObject* img;
    float radius;
    int error;
    if (!PyArg_ParseTuple(args, "O!f", &PyArray_Type, &img, &radius)) {
        return NULL;
    }
    if (CheckImg(img, NPY_FLOAT32)) {
        return NULL;
    }
    if (radius <= 0) {
        PyErr_SetString(PyExc_ValueError, "'radius' has to be > 0");
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    error = CClose(img, radius);
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_RuntimeError, "failed to execute close");
        return NULL;
    }
    Py_INCREF(img);
    return (PyObject *)img;
}


static PyObject* Dilate(PyObject* self, PyObject* args) {
    // Morphological dilatation
    PyArrayObject* img;
    float radius;
    int error;
    if (!PyArg_ParseTuple(args, "O!f", &PyArray_Type, &img, &radius)) {
        return NULL;
    }
    if (CheckImg(img, NPY_FLOAT32)) {
        return NULL;
    }
    if (radius <= 0) {
        PyErr_SetString(PyExc_ValueError, "'radius' has to be > 0");
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    error = Morphology(img, radius, &max);
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_RuntimeError, "failed to execute dilate");
        return NULL;
    }
    Py_INCREF(img);
    return (PyObject *)img;
}


static PyObject* Erode(PyObject* self, PyObject* args) {
    // Morphological erosion
    PyArrayObject* img;
    float radius;
    int error;
    if (!PyArg_ParseTuple(args, "O!f", &PyArray_Type, &img, &radius)) {
        return NULL;
    }
    if (CheckImg(img, NPY_FLOAT32)) {
        return NULL;
    }
    if (radius <= 0) {
        PyErr_SetString(PyExc_ValueError, "'radius' has to be > 0");
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    error = Morphology(img, radius, &min);
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_RuntimeError, "failed to execute erode");
        return NULL;
    }
    Py_INCREF(img);
    return (PyObject *)img;
}


static PyObject* Open(PyObject* self, PyObject* args) {
    // Morphological opening
    PyArrayObject* img;
    float radius;
    int error;
    if (!PyArg_ParseTuple(args, "O!f", &PyArray_Type, &img, &radius)) {
        return NULL;
    }
    if (CheckImg(img, NPY_FLOAT32)) {
        return NULL;
    }
    if (radius <= 0) {
        PyErr_SetString(PyExc_ValueError, "'radius' has to be > 0");
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    error = COpen(img, radius);
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_RuntimeError, "failed to execute open");
        return NULL;
    }
    Py_INCREF(img);
    return (PyObject *)img;
}


static PyObject* DilateKernel(PyObject* self, PyObject* args, PyObject* kwargs) {
    // morphological dilatation
    static char *kwlist[] = {"img", "kernel", "out", NULL};
    PyArrayObject *img, *kernel, *out = NULL;
    int error;

    // Parse and check
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "O!O!|O!", kwlist,
        &PyArray_Type, &img, &PyArray_Type, &kernel, &PyArray_Type, &out)
    ) {
        return NULL;
    }
    if (CheckImg(img, NPY_FLOAT32)) {
        return NULL;
    }
    if (CheckKernel(kernel, NPY_UINT8)) {
        return NULL;
    }
    if (out != NULL) {
        if (CheckImg(out, NPY_FLOAT32)) {
            return NULL;
        }
        if (PyArray_DIM(out, 0) != PyArray_DIM(img, 0) || PyArray_DIM(out, 1) != PyArray_DIM(img, 1)) {
            PyErr_SetString(PyExc_ValueError, "'img' and 'out' have not the same shape");
            return NULL;
        }
    }

    // create output
    if (out == NULL) {
        npy_intp shape[2] = {PyArray_DIM(img, 0), PyArray_DIM(img, 1)};
        out = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
        if (out == NULL) {
            return PyErr_NoMemory();
        }
    } else {
        Py_INCREF(out);
    }

    // apply function
    Py_BEGIN_ALLOW_THREADS
    error = Morpho_1_3(
        (npy_float32* )PyArray_DATA(img),
        (npy_float32* )PyArray_DATA(out),
        (long)PyArray_DIM(img, 0),
        (long)PyArray_DIM(img, 1),
        &max
    );
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_RuntimeError, "failed to execute the morphological process");
        Py_DECREF(out);
        return NULL;
    }

    return (PyObject *)out;
}


static PyMethodDef morphoMethods[] = {
    {"close", Close, METH_VARARGS, "Morphological closing."},
    {"dilate", Dilate, METH_VARARGS, "Morphological dilatation."},
    {"erode", Erode, METH_VARARGS, "Morphological erosion."},
    {"open", Open, METH_VARARGS, "Morphological opening."},
    {"dilate_kernel", (PyCFunction)DilateKernel, METH_VARARGS | METH_KEYWORDS, "Test function."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_morpho = {
    PyModuleDef_HEAD_INIT,
    "morpho",
    "Fast Binary Clustering.",
    -1,
    morphoMethods
};


PyMODINIT_FUNC PyInit_c_morpho(void) {
    import_array();
    if ( PyErr_Occurred() ) {
        return NULL;
    }
    return PyModule_Create(&c_morpho);
}
