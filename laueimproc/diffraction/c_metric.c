/* Distance calculation based on hash table. */

#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


int CartesianToPolar(npy_int32* pol, long* limits, npy_float32* rays, const long n, const float res) {
    // convert into spherical normalized coordinate
    npy_float32 inv_res = 1.0 / res;
    long eta_min, eta_max, phi_min, phi_max;

    if (!n) {
        fprintf(stderr, "at least one element is required to compute polar coordinates\n");
        return 1;
    }

    pol[0] = lroundf(acosf(rays[0]) * inv_res);
    pol[1] = lroundf(atan2f(rays[1], rays[2]) * inv_res);
    eta_min = pol[0], eta_max = pol[0];
    phi_min = pol[1], phi_max = pol[1];

    #pragma omp simd
    for (long i = 0; i < n; ++i) {
        long shift = 3 * i;
        long eta = lroundf(acosf(rays[shift]) * inv_res);
        long phi = lroundf(atan2f(rays[shift + 1], rays[shift + 2]) * inv_res);
        pol[2 * i] = (npy_int32)eta, pol[2 * i + 1] = (npy_int32)phi;

        if (eta < eta_min) {
            eta_min = eta;
        } else if (eta > eta_max) {
            eta_max = eta;
        }
        if (phi < phi_min) {
            phi_min = phi;
        } else if (phi > phi_max) {
            phi_max = phi;
        }
    }

    limits[0] = eta_min, limits[1] = eta_max;
    limits[2] = phi_min, limits[3] = phi_max;

    return 0;
}


int PolarToTable(npy_int32* table, npy_int32* indices, npy_int32* pol, long* limits, const long n) {
    /*
        Write the hash table.

        table of shape (a, b, 2) assumed to ba all zero with:
            * table[a, b, 0] -> start pointor of indice list
            * table[a, b, 1] -> number of elements
        indicies is "suffled" range to match the table.
    */
    long eta_min = limits[0], eta_max = limits[1], phi_min = limits[2], phi_max = limits[3];
    long table_eta = eta_max - eta_min + 1;
    long table_phi = phi_max - phi_min + 1;
    long long table_area = (long long)(table_eta) * (long long)(table_phi);

    // find number of rays per unit (initialization)
    for (long i = 0; i < n; ++i) {
        long long eta = (long long)pol[2 * i] - eta_min, phi = (long long)pol[2 * i + 1] - phi_min;
        ++table[2 * (eta * table_phi + phi)];  // not + 1 for optimization
    }

    // find the address of the start point
    long size = table[0], buff_size;
    table[0] = 0;
    for (long long j = 2; j < 2 * table_area; j += 2) {
        buff_size = table[j];
        table[j] = table[j - 2] + size;  // cumsum
        size = buff_size;
    }

    // fill indices
    for (long i = 0; i < n; ++i) {
        long long eta = (long long)pol[2 * i] - eta_min, phi = (long long)pol[2 * i + 1] - phi_min;
        long shift = 2 * (eta * table_phi + phi);
        indices[table[shift] + table[shift + 1]] = i;
        ++table[shift + 1];
    }

    return 0;
}


inline long lfloorf(const double x) {
    long i = (long)x;
    return i - (i > x);
}


int LinkCloseRaysBrut(
    npy_intp* size, long* link, npy_float32* dist,
    npy_float32* rays, const long n, const float res,
    npy_float32* table_rays, const float table_res,
    npy_int32* table, npy_int32* indices, long* limits
) {
    // find the close pairs
    long eta_min = limits[0], eta_max = limits[1], phi_min = limits[2], phi_max = limits[3];
    long ray_rank = 0;
    long table_phi = phi_max - phi_min + 1;
    npy_float32 inv_table_res = 1.0 / table_res;
    float cos_res = cosf(res);

    for (long ray_index = 0; ray_index < n; ++ray_index) {
        long shift_ray = 3 * ray_index;
        long eta_floor = (long)(acosf(rays[shift_ray]) * inv_table_res);  // floor == cast if positive
        if (eta_floor + 1 < eta_min || eta_floor > eta_max) {
            continue;
        }
        long phi_floor = lfloorf(atan2f(rays[shift_ray + 1], rays[shift_ray + 2]) * inv_table_res);
        if (phi_floor + 1 < phi_min || phi_floor > phi_max) {
            continue;
        }

        int no_pair = 1;
        // for eta, phi in [(eta, phi), (eta, phi+1), (eta+1, phi), (eta+1, phi+1)]:
        for (long eta = MAX(eta_min, eta_floor); eta <= MIN(eta_max, eta_floor + 1); ++eta) {
            for (long phi = MAX(phi_min, phi_floor); phi <= MIN(phi_max, phi_floor + 1); ++phi) {
                // fprintf(stdout, "eta:%ld<=%ld<=%ld, phi:%ld<=%ld<=%ld\n", eta_min, eta, eta_max, phi_min, phi, phi_max);

                long shift_table = 2 * ((eta - eta_min) * table_phi + (phi - phi_min));
                long index = table[shift_table];  // table[eta, phi, 0]
                long n_indices = table[shift_table + 1];  // table[eta, phi, 1]
                for (long shift_indices = index; shift_indices < index + n_indices; ++shift_indices) {
                    // fprintf(stdout, "shift_indices=%ld\n", shift_indices);
                    long shift_table_ray = 3 * indices[shift_indices];
                    // fprintf(stdout, "ray_table=(%f, %f, %f), ray=(%f, %f, %f)\n",
                    //     table_rays[shift_table_ray], table_rays[shift_table_ray + 1], table_rays[shift_table_ray + 2],
                    //     rays[shift_ray], rays[shift_ray + 1], rays[shift_ray + 2]
                    // );
                    npy_float32 buff_dist = (
                        rays[shift_ray] * table_rays[shift_table_ray]
                        + rays[shift_ray + 1] * table_rays[shift_table_ray + 1]
                        + rays[shift_ray + 2] * table_rays[shift_table_ray + 2]
                    );
                    if (buff_dist >= cos_res) {
                        if (no_pair || (buff_dist > dist[ray_rank])) {
                            link[2 * ray_rank] = ray_index;
                            link[2 * ray_rank + 1] = indices[shift_indices];
                            dist[ray_rank] = buff_dist;
                            no_pair = 0;
                        }
                    }
                }
            }
        }
        ray_rank += 1 - no_pair;
    }
    *size = ray_rank;
    return 0;
}


static PyObject* RayToTable(PyObject* self, PyObject* args) {
    PyArrayObject *rays, *table, *indices;
    float res;
    npy_intp shape[3];
    long limits[4];
    npy_int32* pol;
    PyObject* out;

    if (!PyArg_ParseTuple(args, "O!f", &PyArray_Type, &rays, &res)) {
        return NULL;
    }

    // verifications
    if (CheckRays(rays)) {
        return NULL;
    }
    if (res < 2.0 * M_PI / 2147483647.0) {
        PyErr_SetString(PyExc_ValueError, "'res' has to be in ]0, 2*pi/(2**31-1)]");
        return NULL;
    }

    // polar projection
    pol = malloc((long)PyArray_DIM(rays, 0) * 2 * sizeof(*pol));
    if (pol == NULL) {
        return PyErr_NoMemory();
    }
    if (CartesianToPolar(pol, limits, (npy_float32*)PyArray_DATA(rays), (long)PyArray_DIM(rays, 0), res)) {
        free(pol);
        PyErr_SetString(PyExc_RuntimeError, "failed to convert into polar coordinates");
        return NULL;
    }

    // split into hash table
    shape[0] = limits[1] - limits[0] + 1;
    shape[1] = limits[3] - limits[2] + 1;
    shape[2] = 2;
    table = (PyArrayObject *)PyArray_ZEROS(3, shape, NPY_INT32, 0);
    if (table == NULL) {
        free(pol);
        return PyErr_NoMemory();
    }
    shape[0] = PyArray_DIM(rays, 0);
    indices = (PyArrayObject *)PyArray_EMPTY(1, shape, NPY_INT32, 0);
    if (indices == NULL) {
        free(pol);
        Py_DECREF(table);
        return PyErr_NoMemory();
    }
    if (PolarToTable((npy_int32*)PyArray_DATA(table), (npy_int32*)PyArray_DATA(indices), pol, limits, (long)PyArray_DIM(rays, 0))) {
        free(pol);
        Py_DECREF(table);
        Py_DECREF(indices);
        PyErr_SetString(PyExc_RuntimeError, "failed to split polar into hash table");
        return NULL;
    }
    free(pol);

    // pack result
    out = Py_BuildValue("(OO(llll))", table, indices, limits[0], limits[1], limits[2], limits[3]);
    Py_DECREF(table);
    Py_DECREF(indices);
    return out;
}


static PyObject* LinkCloseRays(PyObject* self, PyObject* args) {
    /*
        To each ray of 'rays', associate the closest ray of 'table_rays'

        No check for table_rays, and output of ray_to_table.
    */
    float res, table_res;
    int error;
    long limits[4];
    long* link_data;
    npy_float32* dist_data;
    npy_intp shape[2];
    PyArrayObject *link, *dist, *rays, *table_rays, *table, *indices;
    PyObject* out;

    if (!PyArg_ParseTuple(
        args, "O!fO!f(O!O!(llll))",
        &PyArray_Type, &rays, &res,
        &PyArray_Type, &table_rays, &table_res,
        &PyArray_Type, &table, &PyArray_Type, &indices, &(limits[0]), &(limits[1]), &(limits[2]), &(limits[3])

    )) {
        return NULL;
    }

    // verifications
    if (CheckRays(rays)) {
        return NULL;
    }
    if (res < 2.0 * M_PI / 2147483647.0) {
        PyErr_SetString(PyExc_ValueError, "'res' has to be in ]0, 2*pi/(2**31-1)]");
        return NULL;
    }

    // allocation
    link_data = malloc(2 * PyArray_DIM(rays, 0) * sizeof(*link_data));
    if (link_data == NULL) {
        return PyErr_NoMemory();
    }
    dist_data = malloc(PyArray_DIM(rays, 0) * sizeof(*dist_data));
    if (dist_data == NULL) {
        free(link_data);
        return PyErr_NoMemory();
    }

    // matching
    Py_BEGIN_ALLOW_THREADS
    error = LinkCloseRaysBrut(
        shape, link_data, dist_data,
        (npy_float32*)PyArray_DATA(rays), (long)PyArray_DIM(rays, 0), res,
        (npy_float32*)PyArray_DATA(table_rays), table_res,
        (npy_int32*)PyArray_DATA(table), (npy_int32*)PyArray_DATA(indices), limits
    );
    Py_END_ALLOW_THREADS
    if (error) {
        free(link_data);
        free(dist_data);
        PyErr_SetString(PyExc_RuntimeError, "failed to link the close rays");
        return NULL;
    }

    // realloc
    link_data = realloc(link_data, 2 * shape[1] * sizeof(*link_data));
    dist_data = realloc(dist_data, shape[1] * sizeof(*dist_data));
    shape[1] = 2;
    link = (PyArrayObject *)PyArray_SimpleNewFromData(2, shape, NPY_INT32, link_data);
    if (link == NULL) {
        free(link_data);
        free(dist_data);
        return PyErr_NoMemory();
    }
    PyArray_ENABLEFLAGS(link, NPY_ARRAY_OWNDATA);  // for memory leak
    dist = (PyArrayObject *)PyArray_SimpleNewFromData(2, shape, NPY_INT32, dist_data);
    if (dist == NULL) {
        Py_DECREF(link);
        free(dist_data);
        return PyErr_NoMemory();
    }
    PyArray_ENABLEFLAGS(dist, NPY_ARRAY_OWNDATA);

    // pack result
    out = Py_BuildValue("(OO)", link, dist);
    Py_DECREF(link);
    Py_DECREF(dist);
    return out;
}


static PyMethodDef metricMethods[] = {
    {"ray_to_table", RayToTable, METH_VARARGS, "Projects rays into a hash table."},
    {"link_close_rays", LinkCloseRays, METH_VARARGS, "Associate the closet ray."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_metric = {
    PyModuleDef_HEAD_INIT,
    "metric",
    "Fast distance calculation based on hash table.",
    -1,
    metricMethods
};


PyMODINIT_FUNC PyInit_c_metric(void) {
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&c_metric);
}
