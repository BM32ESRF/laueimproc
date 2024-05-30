/* Multivariate gaussian computing. */

#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include "laueimproc/gmm/c_gmm.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>



static PyMethodDef gmmMethods[] = {
    {"compute_rois_centroid", ComputeRoisCentroid, METH_VARARGS, "Compute the weighted barycenter of each roi."},
    {"compute_rois_inertia", ComputeRoisInertia, METH_VARARGS, "Compute the order 2 momentum centered along the barycenter."},
    {"compute_rois_max", ComputeRoisMax, METH_VARARGS, "Compute the argmax of the intensity and the intensity max of each roi."},
    {"compute_rois_sum", ComputeRoisSum, METH_VARARGS, "Compute the sum of the intensities of each roi."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_gmm = {
    PyModuleDef_HEAD_INIT,
    "gmm",
    "Multivariate gaussian computing.",
    -1,
    gmmMethods
};


PyMODINIT_FUNC PyInit_c_gmm(void) {
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&c_gmm);
}
