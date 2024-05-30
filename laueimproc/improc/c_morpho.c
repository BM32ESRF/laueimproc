/* Morphological operations. */


#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include <limits.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <omp.h>  // for thread and simd
#include <Python.h>
#include <stdbool.h>
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


int Kernel2Shift(PyArrayObject* kernel, long* shift[], long* nb_items, long anchor_i, long anchor_j) {
    // Find the list of the position of all non 0 items.
    const long kernel_height = (long)PyArray_DIM(kernel, 0), kernel_width = (long)PyArray_DIM(kernel, 1);
    long local_nb_items;

    // alloc
    *shift = malloc(2 * kernel_height * kernel_width * sizeof(**shift));
    if (*shift == NULL) {
        fprintf(stderr, "failed malloc kernel\n");
        return 1;
    }

    // anchor
    if (anchor_i == LONG_MIN) {
        anchor_i = (kernel_height - 1) / 2;
    }
    if (anchor_j == LONG_MIN) {
        anchor_j = (kernel_width - 1) / 2;
    }

    // exploration
    local_nb_items = 0;  // required for pragma syntax
    #pragma omp simd collapse(2) reduction(+:local_nb_items)
    for (long i = 0; i < kernel_height; ++i) {
        for (long j = 0; j < kernel_width; ++j) {
            if (*(npy_uint8 *)PyArray_GETPTR2(kernel, i, j)) {
                (*shift)[2 * local_nb_items] = i - anchor_i;
                (*shift)[2 * local_nb_items + 1] = j - anchor_j;
                ++local_nb_items;
            }
        }
    }
    *nb_items = local_nb_items;

    // verification
    if (*nb_items == 0) {
        free(*shift);
        fprintf(stderr, "the kernel must contain at least one non zero element\n");
        return 1;
    }

    return 0;
}


#pragma omp declare simd
npy_float32 GeneralMorphoOneItemCheck(
    tqdm img, const long height, const long width,
    const long i, const long j, long* shift, const long nb_items,
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)
) {
    // Combine the elements
    // Check the padding
    npy_float32 combi = 0.0;  // pad with 0 if dilatation or erosion `combi = 1.0 - comp(0.0, 1.0)`
    bool is_set = false;
    long h, w;
    for (long k = 0; k < 2 * nb_items; k += 2) {
        h = i + shift[k];
        if (h < 0 || h >= height) continue;
        w = j + shift[k + 1];
        if (w < 0 || w >= width) continue;
        if (is_set) {
            combi = comp(combi, img[w + width*h]);
        } else {
            combi = img[w + width*h];
            is_set = true;
        }
    }
    return combi;
}


int MorphoBorders(
    tqdm img, tqdm out, const long height, const long width,
    long* shift, const long nb_items,
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2),
    long* i_min_p, long* j_min_p, long* i_max_p, long* j_max_p
) {
    // Naive morpho on the borders of the picture.
    // Update ij_min and ij_max such as ij_min <= ij < ij_max is padding safe
    long i = 0, h, w;
    // bool padding;

    // find and set padding limits
    *i_min_p = 0, *j_min_p = 0;
    *i_max_p = height, *j_max_p = width;
    for (long k = 0; k < 2 * nb_items; k += 2) {
        h = -shift[k], w = -shift[k + 1];
        if (h > *i_min_p) *i_min_p = h;
        if (w > *j_min_p) *j_min_p = w;
        h = height + h, w = width + w;
        if (h < *i_max_p) *i_max_p = h;
        if (w < *j_max_p) *j_max_p = w;
    }

    // resolve overlapp
    if (*i_max_p < *i_min_p) {
        *i_max_p = *i_min_p;
    }
    if (*j_max_p < *j_min_p) {
        *j_max_p = *j_min_p;
    }

    // fprintf(stdout, "padding safe area: %ld<=i<%ld and %ld<=j<%ld\n", *i_min_p, *i_max_p, *j_min_p, *j_max_p);

    // fill borders
    for (/*long i = 0*/; i < *i_min_p; ++i) {
        // top
        for (long j = 0; j < width; ++j) {
            out[j + width*i] = GeneralMorphoOneItemCheck(img, height, width, i, j, shift, nb_items, comp);
        }
    }
    for (/*long i = *i_min_p*/; i < *i_max_p; ++i) {
        // middle left band
        for (long j = 0; j < *j_min_p; ++j) {
            out[j + width*i] = GeneralMorphoOneItemCheck(img, height, width, i, j, shift, nb_items, comp);
        }
        // middle right band
        for (long j = *j_max_p; j < width; ++j) {
            out[j + width*i] = GeneralMorphoOneItemCheck(img, height, width, i, j, shift, nb_items, comp);
        }
    }
    for (/*long i = *i_max_p*/; i < height; ++i) {
        // bottom
        for (long j = 0; j < width; ++j) {
            out[j + width*i] = GeneralMorphoOneItemCheck(img, height, width, i, j, shift, nb_items, comp);
        }
    }

    return 0;
}


int MorphoMiddle(
    tqdm img, tqdm out, const long width,
    long* shift, const long nb_items,
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2),
    const long i_min, const long j_min, const long i_max, const long j_max
) {
    // Apply the morphological operation in the area i_min <= i < i_max and j_min <= j < j_max.
    // In this area, it is assumed that no padding is required.

    #pragma omp parallel for simd schedule(static)
    for (long i = i_min; i < i_max; ++i) {
        // long i_s = width * i;
        // long h_s = width * (i + shift[0]);
        // memcpy(out + j_min + i_s, img + j_min + shift[1] + h_s, (j_max - j_min) * sizeof(*img));
        // for (long k = 2; k < 2 * nb_items; k += 2) {
        //     long h_s = width * (i + shift[k]);
        //     for (long j = j_min; j < j_max; ++j) {
        //         long w = j + shift[k + 1];
        //         out[j + i_s] = comp(out[j + i_s], img[w + h_s]);
        //     }
        // }
        for (long j = j_min; j < j_max; ++j) {
            npy_float32 combi;
            long h = i + shift[0], w = j + shift[1];
            combi = img[w + width*h];
            for (long k = 2; k < 2 * nb_items; k += 2) {
                h = i + shift[k], w = j + shift[k + 1];
                // combi = comp(combi, img[w + width*h]);
                combi = max(combi, img[w + width*h]);
            }
            out[j + width*i] = combi;
        }
    }
    return 0;
}


int MorphoApply(
    tqdm img, tqdm out, const long height, const long width,
    long* shift, const long nb_items,
    npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)
) {
    // General morpho on any kernel.
    long i_min, j_min, i_max, j_max;
    int error;

    // borders (with padding check)
    error = MorphoBorders(
        img, out, height, width,
        shift, nb_items,
        comp,
        &i_min, &j_min, &i_max, &j_max
    );

    // middle (no check)
    if (!error && i_max > i_min && j_max > j_min) {
        return MorphoMiddle(
            img, out, width,
            shift, nb_items,
            comp,
            i_min, j_min, i_max, j_max
        );
    }

    return error;
}


int UpdateLine(
    tqdm src,  // the source lines
    tqdm dst,  // the destination lines
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
    tqdm img,  // src image
    tqdm dst,  // data to share comparisons result
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


int Morphology(PyArrayObject* img, const float radius, npy_float32 (*comp)(const npy_float32 val1, const npy_float32 val2)) {
    /* Apply the morphological simple operation (func comp) on the image img with a circular kernel */

    int error;
    const long height = (long)PyArray_DIM(img, 0), width = (long)PyArray_DIM(img, 1);
    const long layer_stride = height * width;
    tqdm img_data = (tqdm )PyArray_DATA(img);
    tqdm layers;
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
    static char *kwlist[] = {"img", "kernel", "out", "anchor", NULL};
    PyArrayObject *img, *kernel, *out = NULL;
    long* shift;  // coordinate representation of the kernel
    long nb_items;  // nbr of non zero item in kernel
    long anchor_i = LONG_MIN, anchor_j = LONG_MIN;
    int error;

    // Parse and basic check
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "O!O!|O!(ll)", kwlist,
        &PyArray_Type, &img, &PyArray_Type, &kernel, &PyArray_Type, &out, &anchor_i, &anchor_j)
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

    // kernel patch to kernel index
    Py_BEGIN_ALLOW_THREADS
    error = Kernel2Shift(kernel, &shift, &nb_items, anchor_i, anchor_j);
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_RuntimeError, "failed to rewrite the kernel");
        Py_DECREF(out);
        return NULL;
    }

    // apply morpho operation
    Py_BEGIN_ALLOW_THREADS
    error = MorphoApply(
        (tqdm )PyArray_DATA(img), (tqdm )PyArray_DATA(out), (long)PyArray_DIM(img, 0), (long)PyArray_DIM(img, 1),
        shift, nb_items,
        &max
    );
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_RuntimeError, "failed to execute the morphological process");
        free(shift);
        Py_DECREF(out);
        return NULL;
    }

    // // apply function
    // Py_BEGIN_ALLOW_THREADS
    // error = Morpho_1_3(
    //     (tqdm )PyArray_DATA(img),
    //     (tqdm )PyArray_DATA(out),
    //     (long)PyArray_DIM(img, 0),
    //     (long)PyArray_DIM(img, 1),
    //     &max
    // );
    // Py_END_ALLOW_THREADS
    // if (error) {
    //     PyErr_SetString(PyExc_RuntimeError, "failed to execute the morphological process");
    //     free(shift);
    //     Py_DECREF(out);
    //     return NULL;
    // }

    free(shift);
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
