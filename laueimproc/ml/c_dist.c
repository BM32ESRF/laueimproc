/* Distance calculation. */

#define PY_SSIZE_T_CLEAN
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int TestVectorTuple(PyObject* tuple, const long dim, npy_float32 *point_data[]) {
    // Check if tuple if a d len vector of scalars, fill the c scalar vector.
    if ((long)PyTuple_GET_SIZE(tuple) != dim) {
        fprintf(stderr, "error info: dim=%ld vs len(tuple)=%ld\n", dim, (long)PyTuple_GET_SIZE(tuple));
        PyErr_SetString(PyExc_ValueError, "the numbers of dimensions if not the same");
        return 1;
    }
    *point_data = malloc(dim * sizeof(**point_data));
    if (*point_data == NULL) {
        PyErr_SetString(PyExc_MemoryError, "failed to malloc");
        return 1;
    }
    for (long j = 0; j < dim; ++j) {
        (*point_data)[j] = PyFloat_AsDouble(PyTuple_GET_ITEM(tuple, (Py_ssize_t)(j)));
        if ((*point_data)[j] == -1.0 && PyErr_Occurred()) {
            return 1;
        }
    }
    return 0;
}


#pragma omp declare simd
npy_float32 square(npy_float32 x) {
    return x * x;
}

#pragma omp declare simd
npy_float32 dist_square(npy_float32 x, npy_float32 p) {
    npy_float32 d;
    d = x - p;
    d *= d;
    return d;
}

#pragma omp declare simd
npy_float32 dist_square_scaled(npy_float32 x, npy_float32 p, npy_float32 s) {
    npy_float32 d;
    d = x - p;
    d *= s;
    d *= d;
    return d;
}

int check_in_ball(npy_float32* coords_i, npy_float32* point_data, npy_float32* tol_data, const long dim) {
    // return 0 if the point is in the tolerence range, 1 otherwise
    for (long j = 0; j < dim; ++j) {
        if (fabs(coords_i[j] - point_data[j]) > tol_data[j]) {
            return 1;
        }
    }
    return 0;
}


int SelectClosetPointC(
    long* argmin,
    npy_float32* coords, npy_float32* point_data, npy_float32* tol_data, npy_float32* scale_data,
    const long n, const long dim
) {
    // Exclude the points to far from the destination, keep the closet.
    long is;
    npy_float32 dist, best_dist = 0.0;

    if (n == 0) {
        return 1;
    }

    if (tol_data == NULL) {  // case no point to exclude
        *argmin = 0;
        if (scale_data == NULL) {  // no convertion factor
            // first loop
            #pragma omp simd reduction(+:best_dist)
            for (long j = 0; j < dim; ++j) {
                best_dist += dist_square(point_data[j], coords[j]);
            }
            // next loops
            for (long i = 1; i < n; ++i) {
                is = i * dim;
                dist = 0.0;
                #pragma omp simd reduction(+:dist)
                for (long j = 0; j < dim; ++j) {
                    dist += dist_square(point_data[j], coords[is + j]);
                }
                if (dist < best_dist) {
                    best_dist = dist, *argmin = i;
                }
            }
        } else {  // convertion factor
            // first loop
            #pragma omp simd reduction(+:best_dist)
            for (long j = 0; j < dim; ++j) {
                best_dist += dist_square_scaled(point_data[j], coords[j], scale_data[j]);
            }
            // next loops
            for (long i = 1; i < n; ++i) {
                is = i * dim;
                dist = 0.0;
                #pragma omp simd reduction(+:dist)
                for (long j = 0; j < dim; ++j) {
                    dist += dist_square_scaled(point_data[j], coords[is + j], scale_data[j]);
                }
                if (dist < best_dist) {
                    best_dist = dist, *argmin = i;
                }
            }
        }
    } else {  // case point to exclude
        *argmin = -1;
        if (scale_data == NULL) {  // no convertion factor
            for (long i = 0; i < n; ++i) {
                is = i * dim;
                if (check_in_ball(coords+is, point_data, tol_data, dim)) {
                    continue;
                }
                dist = 0.0;
                #pragma omp simd reduction(+:dist)
                for (long j = 0; j < dim; ++j) {
                    dist += dist_square(point_data[j], coords[is + j]);
                }
                if (*argmin == -1 || dist < best_dist) {
                    best_dist = dist, *argmin = i;
                }
            }
        } else {  // convertion factor
            for (long i = 0; i < n; ++i) {
                is = i * dim;
                if (check_in_ball(coords+is, point_data, tol_data, dim)) {
                    continue;
                }
                dist = 0.0;
                #pragma omp simd reduction(+:dist)
                for (long j = 0; j < dim; ++j) {
                    dist += dist_square_scaled(point_data[j], coords[is + j], scale_data[j]);
                }
                if (*argmin == -1 || dist < best_dist) {
                    best_dist = dist, *argmin = i;
                }
            }
        }
        if (*argmin == -1) {
            return 1;
        }
    }
    return 0;
}


static PyObject* SelectClosestPoint(PyObject* self, PyObject* args, PyObject* kwargs) {
    // Find the closest point.
    static char *kwlist[] = {"coords", "point", "tol", "scale", NULL};
    PyArrayObject *coords;
    PyObject *point, *tol=NULL, *scale=NULL;
    npy_float32 *point_data, *tol_data=NULL, *scale_data=NULL;
    long n, dim, argmin;
    int error;

    // parse and check
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "O!O!|O!O!", kwlist,
        &PyArray_Type, &coords, &PyTuple_Type, &point, &PyTuple_Type, &tol, &PyTuple_Type, &scale)
    ) {
        return NULL;
    }
    if (PyArray_NDIM(coords) != 2) {
        PyErr_SetString(PyExc_ValueError, "'coords' requires 2 dimensions");
        return NULL;
    }
    if (PyArray_TYPE(coords) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "'coords' has to be of type float32");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(coords)) {
        PyErr_SetString(PyExc_ValueError, "'coords' has to be c contiguous");
        return NULL;
    }
    dim = (long)PyArray_DIM(coords, 1);
    if (TestVectorTuple(point, dim, &point_data)) {
        return NULL;
    }
    if (tol != NULL && TestVectorTuple(tol, dim, &tol_data)) {
        free(point_data);
        return NULL;
    }
    if (scale != NULL && TestVectorTuple(scale, dim, &scale_data)) {
        free(point_data);
        free(tol_data);
        return NULL;
    }

    // find closet
    Py_BEGIN_ALLOW_THREADS
    n = (long)PyArray_DIM(coords, 0);
    error = SelectClosetPointC(&argmin, (npy_float32 *)PyArray_DATA(coords), point_data, tol_data, scale_data, n, dim);
    Py_END_ALLOW_THREADS
    if (error) {
        free(point_data);
        free(tol_data);
        free(scale_data);
        PyErr_SetString(PyExc_LookupError, "no point match");
        return NULL;
    }

    // parse result
    free(point_data);
    free(tol_data);
    free(scale_data);
    return Py_BuildValue("l", argmin);
}


static PyMethodDef distMethods[] = {
    {
        "select_closest_point",
        (PyCFunction)SelectClosestPoint,
        METH_VARARGS | METH_KEYWORDS,
        "Select the closest point according the manhattan tolerancy and an euclidian distance."
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_dist = {
    PyModuleDef_HEAD_INIT,
    "dist",
    "Distance calculation.",
    -1,
    distMethods
};


PyMODINIT_FUNC PyInit_c_dist(void) {
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&c_dist);
}
