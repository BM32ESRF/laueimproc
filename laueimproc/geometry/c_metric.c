/* Distance calculation based on hash table. */

#define PY_SSIZE_T_CLEAN
#include "laueimproc/c_check.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


int CartesianToSpherical(npy_int32* pol, long* limits, npy_float32* rays, const long n, const float res) {
    /*
        Convertion of 3d unitary vectors into pseudo spherical coordinates.

        Let u be the vector (x, y, z).
        We have eta = acos(x) in [0, pi].
        We should have phi = atan2(y, z), but to avoid the discontinuity in -pi = +pi,
        we use the non bijective phi = atan2(|y|, z) in [0, pi] as well.
    */
    long eta_min, eta_max, phi_min, phi_max;
    npy_float32 inv_res = 1.0 / res;

    if (!n) {
        fprintf(stderr, "at least one element is required to compute polar coordinates\n");
        return 1;
    }

    // initialisation for min max
    pol[0] = lroundf(inv_res * acosf(rays[0]));
    pol[1] = lroundf(inv_res * atan2f(fabsf(rays[1]), rays[2]));
    eta_min = pol[0], eta_max = pol[0];
    phi_min = pol[1], phi_max = pol[1];

    #pragma omp simd
    for (long i = 1; i < n; ++i) {
        // compute spherical
        long shift = 3 * i;
        long eta = lroundf(inv_res * acosf(rays[shift]));
        long phi = lroundf(inv_res * atan2f(fabsf(rays[shift + 1]), rays[shift + 2]));
        pol[2 * i] = (npy_int32)eta, pol[2 * i + 1] = (npy_int32)phi;

        // update min max
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

    // write bounds
    limits[0] = eta_min, limits[1] = eta_max;
    limits[2] = phi_min, limits[3] = phi_max;

    return 0;
}


int SphericalToTable(npy_int32* table, npy_int32* indices, npy_int32* pol, long* limits, const long n) {
    /*
        Write the hash table.

        table of shape (a, b, 2) assumed to ba all zero with:
            * table[a, b, 0] -> start pointor of indice list
            * table[a, b, 1] -> number of elements
        indices is "suffled" range to match the table.
    */
    long long eta_min = limits[0], eta_max = limits[1], phi_min = limits[2], phi_max = limits[3];
    long long table_eta = eta_max - eta_min + 1;
    long long table_phi = phi_max - phi_min + 1;
    long long table_area = (long long)(table_eta) * (long long)(table_phi);

    // find number of rays per unit (initialization)
    for (long long i = 0; i < n; ++i) {
        long long eta = (long long)pol[2 * i] - eta_min, phi = (long long)pol[2 * i + 1] - phi_min;
        ++table[2 * (eta * table_phi + phi)];  // not + 1 for optimization
    }

    // find the address of the start point
    long long size = table[0], buff_size;
    table[0] = 0;
    for (long long j = 2; j < 2 * table_area; j += 2) {
        buff_size = table[j];
        table[j] = table[j - 2] + size;  // cumsum
        size = buff_size;
    }

    // fill indices
    for (long long i = 0; i < n; ++i) {
        long long eta = (long long)pol[2 * i] - eta_min, phi = (long long)pol[2 * i + 1] - phi_min;
        long long shift = 2 * (eta * table_phi + phi);
        indices[table[shift] + table[shift + 1]] = i;
        ++table[shift + 1];
    }

    return 0;
}


int CountMatchingRays(
    long* rate,
    npy_float32* rays, const long n, const float res,
    npy_float32* table_rays, const float table_res,
    npy_int32* table, npy_int32* indices, long* limits
) {
    /*
        Count the number of ray in `rays`, close enouth to a ray of `table_rays`.
    */
    long eta_min = limits[0], eta_max = limits[1], phi_min = limits[2], phi_max = limits[3];
    npy_intp ray_rank = 0;
    long table_phi = phi_max - phi_min + 1;
    npy_float32 inv_table_res = 1.0 / table_res;
    float cos_res = cosf(res);

    for (long ray_index = 0; ray_index < n; ++ray_index) {
        // convertion into pseudo spherical
        long shift_ray = 3 * ray_index;
        long eta_floor = (long)(inv_table_res * acosf(rays[shift_ray]));  // cast == floor if positive
        if (eta_floor + 1 < eta_min || eta_floor > eta_max) {
            continue;
        }
        long phi_floor = (long)(inv_table_res * atan2f(fabsf(rays[shift_ray + 1]), rays[shift_ray + 2]));
        if (phi_floor + 1 < phi_min || phi_floor > phi_max) {
            continue;
        }

        // explore neighbours
        int no_pair = 1;
        for (long eta = MAX(eta_min, eta_floor); no_pair && eta <= MIN(eta_max, eta_floor + 1); ++eta) {
            for (long phi = MAX(phi_min, phi_floor); no_pair && phi <= MIN(phi_max, phi_floor + 1); ++phi) {
                long shift_table = 2 * ((eta - eta_min) * table_phi + (phi - phi_min));
                long index = table[shift_table];  // table[eta, phi, 0]
                long n_indices = table[shift_table + 1];  // table[eta, phi, 1]
                for (long shift_indices = index; shift_indices < index + n_indices; ++shift_indices) {
                    long shift_table_ray = 3 * indices[shift_indices];
                    npy_float32 buff_dist = (  // compute dist: cos(theta) = <u1, u2>
                        rays[shift_ray] * table_rays[shift_table_ray]
                        + rays[shift_ray + 1] * table_rays[shift_table_ray + 1]
                        + rays[shift_ray + 2] * table_rays[shift_table_ray + 2]
                    );
                    if (buff_dist >= cos_res) {  // theta <= theta_max <=> cos(theta) >= cos(theta_max)
                        ++ray_rank;
                        no_pair = 0;
                        break;
                    }
                }
            }
        }
    }
    *rate = ray_rank;
    return 0;
}


int LinkCloseRaysBrut(
    npy_intp* size, npy_int32* link,
    npy_float32* rays, const long n, const float res,
    npy_float32* table_rays, const float table_res,
    npy_int32* table, npy_int32* indices, long* limits
) {
    /*
        Compute sparse angular distace matrix.

        For each ray of `rays`, find the closet ray of `table_rays`.
        If the angular distance between this to rays is <= `res`,
        the couple of incicies (index_ray, index_table_ray) is append
        to the `link` list. The scalar product of these two vectors is append to th list `dist`.
    */
    long eta_min = limits[0], eta_max = limits[1], phi_min = limits[2], phi_max = limits[3];
    npy_intp ray_rank = 0;
    long table_phi = phi_max - phi_min + 1;
    npy_float32 inv_table_res = 1.0 / table_res;
    float cos_res = cosf(res);

    for (long ray_index = 0; ray_index < n; ++ray_index) {
        // convertion into pseudo spherical
        long shift_ray = 3 * ray_index;
        long eta_floor = (long)(inv_table_res * acosf(rays[shift_ray]));  // cast == floor if positive
        if (eta_floor + 1 < eta_min || eta_floor > eta_max) {
            continue;
        }
        long phi_floor = (long)(inv_table_res * atan2f(fabsf(rays[shift_ray + 1]), rays[shift_ray + 2]));
        if (phi_floor + 1 < phi_min || phi_floor > phi_max) {
            continue;
        }

        // explore neighbours
        int has_pair = 0;
        npy_float32 max_dist = 0.0;
        for (long eta = MAX(eta_min, eta_floor); eta <= MIN(eta_max, eta_floor + 1); ++eta) {
            for (long phi = MAX(phi_min, phi_floor); phi <= MIN(phi_max, phi_floor + 1); ++phi) {
                long shift_table = 2 * ((eta - eta_min) * table_phi + (phi - phi_min));
                long index = table[shift_table];  // table[eta, phi, 0]
                long n_indices = table[shift_table + 1];  // table[eta, phi, 1]
                for (long shift_indices = index; shift_indices < index + n_indices; ++shift_indices) {
                    long shift_table_ray = 3 * indices[shift_indices];
                    npy_float32 buff_dist = (  // compute dist: cos(theta) = <u1, u2>
                        rays[shift_ray] * table_rays[shift_table_ray]
                        + rays[shift_ray + 1] * table_rays[shift_table_ray + 1]
                        + rays[shift_ray + 2] * table_rays[shift_table_ray + 2]
                    );
                    if (buff_dist >= cos_res) {  // theta <= theta_max <=> cos(theta) >= cos(theta_max)
                        if (!has_pair || (buff_dist > max_dist)) {
                            link[2 * ray_rank] = (npy_int32)ray_index;
                            link[2 * ray_rank + 1] = (npy_int32)indices[shift_indices];
                            max_dist = buff_dist;
                            has_pair = 1;
                        }
                    }
                }
            }
        }
        ray_rank += has_pair;
    }
    *size = ray_rank;
    return 0;
}


static PyObject* RayToTable(PyObject* self, PyObject* args) {
    /*
        Return the (a, b, 2) hash table, the (n,) indices list associated to the hash table and the limits.
    */
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
    if (CartesianToSpherical(pol, limits, (npy_float32*)PyArray_DATA(rays), (long)PyArray_DIM(rays, 0), res)) {
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
    if (SphericalToTable((npy_int32*)PyArray_DATA(table), (npy_int32*)PyArray_DATA(indices), pol, limits, (long)PyArray_DIM(rays, 0))) {
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


static PyObject* MatchingRate(PyObject* self, PyObject* args) {
    /*
        Count the number of ray in `rays`, close enouth to a ray of `table_rays`.

        No check for table_rays, and output of ray_to_table.
    */
    float res, table_res;
    int error;
    long limits[4];
    PyArrayObject *rays, *table_rays, *table, *indices;
    long rate;

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
    if (res >= 2.0 * table_res) {
        PyErr_SetString(PyExc_ValueError, "the table_res has to be stricly grater than half res");
        return NULL;
    }

    // matching
    Py_BEGIN_ALLOW_THREADS
    error = CountMatchingRays(
        &rate,
        (npy_float32*)PyArray_DATA(rays), (long)PyArray_DIM(rays, 0), res,
        (npy_float32*)PyArray_DATA(table_rays), table_res,
        (npy_int32*)PyArray_DATA(table), (npy_int32*)PyArray_DATA(indices), limits
    );
    Py_END_ALLOW_THREADS
    if (error) {
        PyErr_SetString(PyExc_RuntimeError, "failed to link the close rays");
        return NULL;
    }

    // pack result
    return Py_BuildValue("l", rate);
}


static PyObject* LinkCloseRays(PyObject* self, PyObject* args) {
    /*
        To each ray of 'rays', associate the closest ray of 'table_rays'

        No check for table_rays, and output of ray_to_table.
    */
    float res, table_res;
    int error;
    long limits[4];
    npy_int32* link_data;
    npy_intp shape[2];
    PyArrayObject *link, *rays, *table_rays, *table, *indices;

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
    if (res >= 2.0 * table_res) {
        PyErr_SetString(PyExc_ValueError, "the table_res has to be stricly grater than half res");
        return NULL;
    }

    // allocation
    link_data = malloc(2 * PyArray_DIM(rays, 0) * sizeof(*link_data));
    if (link_data == NULL) {
        return PyErr_NoMemory();
    }

    // matching
    Py_BEGIN_ALLOW_THREADS
    error = LinkCloseRaysBrut(
        shape, link_data,
        (npy_float32*)PyArray_DATA(rays), (long)PyArray_DIM(rays, 0), res,
        (npy_float32*)PyArray_DATA(table_rays), table_res,
        (npy_int32*)PyArray_DATA(table), (npy_int32*)PyArray_DATA(indices), limits
    );
    Py_END_ALLOW_THREADS
    if (error) {
        free(link_data);
        PyErr_SetString(PyExc_RuntimeError, "failed to link the close rays");
        return NULL;
    }

    // realloc
    link_data = (npy_int32 *)realloc(link_data, 2 * shape[0] * sizeof(*link_data));
    shape[1] = 2;
    link = (PyArrayObject *)PyArray_SimpleNewFromData(2, shape, NPY_INT32, link_data);
    if (link == NULL) {
        free(link_data);
        return PyErr_NoMemory();
    }
    PyArray_ENABLEFLAGS(link, NPY_ARRAY_OWNDATA);  // for memory leak

    // pack result
    return (PyObject *)(link);
}


static PyMethodDef metricMethods[] = {
    {"ray_to_table", RayToTable, METH_VARARGS, "Projects rays into a hash table."},
    {"matching_rate", MatchingRate, METH_VARARGS, "Count the number of ray close enouth to any reference ray."},
    {"link_close_rays", LinkCloseRays, METH_VARARGS, "Associate the closest ray."},
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
