/* Fast computation of PCA. */

#define PY_SSIZE_T_CLEAN
#include <math.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>

#define EPS 1.1920929e-7f

int checkShapes(PyArrayObject *shapes) {
    // Raise an exception if shape format is not correct.
    if ( PyArray_NDIM(shapes) != 2 ) {
        PyErr_SetString(PyExc_ValueError, "'shapes' requires 2 dimensions");
        return 1;
    }
    if ( PyArray_DIM(shapes, 1) != 2 ) {
        PyErr_SetString(PyExc_ValueError, "second axis of 'shapes' has to be of size 2, for height and width");
        return 1;
    }
    if ( PyArray_TYPE(shapes) != NPY_INT32 ) {
        PyErr_SetString(PyExc_TypeError, "'shapes' has to be of type int32");
        return 1;
    }
    return 0;
}


int computeSinglePCA(npy_float *rawout, npy_float *weights, const npy_int32 height, const npy_int32 width) {
    // compute one PCA, store std1, std2 and theta in rawout
    npy_float weight, norm, mu_h, mu_w;
    npy_float sigma1, sigma2, corr, i_cent, j_cent;
    npy_int32 i, j, k;

    // get mean and total intensity
    norm = 0, mu_h = 0, mu_w = 0;
    for ( i = 0; i < height; ++i ) {
        k = i * width;
        for ( j = 0; j < width; ++j ) {
            weight = *(weights + k + j);
            norm += weight;
            mu_h += weight * (npy_float)i;
            mu_w += weight * (npy_float)j;
        }
    }
    if ( !norm ) {
        fprintf(stderr, "failed to compute pca because all the dup_w are equal to 0\n");
        return 1;
    }
    norm = 1 / norm;
    mu_h *= norm;
    mu_w *= norm;

    // get cov matrix
    sigma1 = 0, sigma2 = 0, corr = 0;
    for ( i = 0; i < height; ++i ) {
        k = i * width;
        i_cent = (npy_float)i - mu_h;
        for ( j = 0; j < width; ++j ) {
            weight = *(weights + k + j);
            weight *= weight;
            j_cent = (npy_float)j - mu_w;
            sigma1 += weight * i_cent * i_cent;
            sigma2 += weight * j_cent * j_cent;
            corr += weight * i_cent * j_cent;
        }
    }
    norm *= norm;
    sigma1 *= norm;
    sigma2 *= norm;
    corr *= norm;

    // diagonalization
    corr *= 2;  // 2*c
    sigma2 += sigma1;  // s1+s2
    sigma1 = sigma2 - 2*sigma1;  // s2-s1
    norm = corr*corr + sigma1*sigma1;  // (2*c)**2 + (s2-s1)**2
    norm = sqrtf( norm );  // sqrt((2*c)**2 + (s2-s1)**2)
    if ( fabsf( corr ) > EPS ) {
        sigma1 += norm;  // s2 - s1 + sqrt((2*c)**2 + (s2-s1)**2)
        sigma1 /= corr;  // (s2 - s1 + sqrt((2*c)**2 + (s2-s1)**2)) / (2*c)
        rawout[2] = atanf( sigma1 );  // theta
    }
    else {
        rawout[2] = sigma1 > EPS ? 0.5*M_PI : 0;  // theta
    }
    sigma1 = sigma2 - norm;  // s1 + s2 - sqrt((2*c)**2 + (s2-s1)**2)
    sigma2 += norm;  // s1 + s2 + sqrt((2*c)**2 + (s2-s1)**2)
    sigma1 *= 0.5;  // lambda2 = 1/2 * (s1 + s2 - sqrt((2*c)**2 + (s2-s1)**2))
    sigma2 *= 0.5;  // lambda1 = 1/2 * (s1 + s2 + sqrt((2*c)**2 + (s2-s1)**2))
    rawout[1] = sqrtf( sigma1 );  // std2
    rawout[0] = sqrtf( sigma2 );  // std1
    return 0;
}


int computeAllPCA(npy_float *rawout, npy_float *weights, const Py_ssize_t datalen, PyArrayObject *shapes) {
    const npy_intp n = PyArray_DIM(shapes, 0);
    npy_intp i;
    npy_int32 height, width, area;
    Py_ssize_t shift = 0;
    for ( i = 0; i < n; ++i ) {
        height = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 0);
        width = *(npy_int32 *)PyArray_GETPTR2(shapes, i, 1);
        area = height * width;
        if ( !area ) {
            fprintf(stderr, "the bbox %ld has zero area\n", i);
            return 1;
        }
        if ( shift + area > datalen ) {
            fprintf(stderr, "the data length 4*%ld is too short to fill roi of index %ld\n", datalen, i);
            return 1;
        }
        if ( computeSinglePCA(rawout+3*i, weights+shift, height, width) ) {
            return 1;
        }
        shift += area;
    }
    if ( datalen != shift ) {  // data to long
        fprintf(stderr, "the data length 4*%ld is too long to fill a total area of %ld pxls.\n", datalen, shift);
        return 1;
    }
    return 0;
}


static PyObject *pca(PyObject *self, PyObject *args) {
    // compute pca on each spot
    PyArrayObject *out, *shapes;
    npy_intp shape[2];
    PyByteArrayObject *data;
    npy_float *rawout, *weights;
    Py_ssize_t datalen;
    int error;

    if ( !PyArg_ParseTuple(args, "YO!", &data, &PyArray_Type, &shapes) ) {
        return NULL;
    }
    weights = (npy_float *)PyByteArray_AsString((PyObject *)data);

    // verifications
    if ( PyByteArray_Size((PyObject *)data) % sizeof(npy_float) ) {
        PyErr_SetString(PyExc_ValueError, "data length is not a multiple of float32 length");
        return NULL;
    }
    if ( NULL == weights ){
        PyErr_SetString(PyExc_ValueError, "data is empty");
        return NULL;
    }
    if ( checkShapes(shapes) ) {
        return NULL;
    }

    // create output array
    shape[0] = PyArray_DIM(shapes, 0);
    shape[1] = 3;
    out = (PyArrayObject *)PyArray_EMPTY(2, shape, NPY_FLOAT32, 0);  // c contiguous
    if ( NULL == out ) {
        PyErr_SetString(PyExc_RuntimeError, "failed to create the output array");
        return NULL;
    }
    rawout = (npy_float *)PyArray_DATA(out);

    // compute pca
    Py_BEGIN_ALLOW_THREADS
    datalen = PyByteArray_Size((PyObject *)data) / sizeof(npy_float);
    error = computeAllPCA(rawout, weights, datalen, shapes);
    Py_END_ALLOW_THREADS
    if ( error ) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "failed to compute the pca");
        return NULL;
    }

    return (PyObject *)out;
}


static PyMethodDef pcaMethods[] = {
    {"pca", pca, METH_VARARGS, "Compute the PCA for each spot."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_pca = {
    PyModuleDef_HEAD_INIT,
    "pca",
    "Fast Principal Componant Annalysis Clculus.",
    -1,
    pcaMethods
};


PyMODINIT_FUNC PyInit_c_pca(void) {
    import_array();
    if ( PyErr_Occurred() ) {
        return NULL;
    }
    return PyModule_Create(&c_pca);
}
