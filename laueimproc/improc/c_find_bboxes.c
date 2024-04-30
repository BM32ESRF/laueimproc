/* Find the bboxes of a binary image. */


#define PY_SSIZE_T_CLEAN
#include <laueimproc/c_check.h>
#include <laueimproc/improc/spot/c_spot_apply.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>
// #include <stdlib.h>


int MergeBBoxes(const npy_int16* bbox_src, npy_int16* bbox_dst) {
    // bbox_dst = union(bbox_dst, bbox_src)
    npy_int16 sum[2];

    sum[0] = bbox_src[0] + bbox_src[2], sum[1] = bbox_dst[0] + bbox_dst[2];  // SIMD
    bbox_dst[0] = bbox_src[0] < bbox_dst[0] ? bbox_src[0] : bbox_dst[0];  // min
    bbox_dst[2] = (sum[0] > sum[1] ? sum[0] : sum[1]) - bbox_dst[0];

    sum[0] = bbox_src[1] + bbox_src[3], sum[1] = bbox_dst[1] + bbox_dst[3];  // SIMD
    bbox_dst[1] = bbox_src[1] < bbox_dst[1] ? bbox_src[1] : bbox_dst[1];  // min
    bbox_dst[3] = (sum[0] > sum[1] ? sum[0] : sum[1]) - bbox_dst[1];

    return 0;
}


int FindClustersAndBBoxes(
    npy_int16** bboxes_p,
    npy_intp* n,
    const unsigned char* binary,
    const unsigned long height,
    const unsigned long width,
    const npy_int16 max_size
) {
    unsigned long* clusters;
    npy_int16* merge;
    unsigned long curr_clus = 0;
    unsigned long i, j, clus_top, clus_left, shift_img, stride;
    const unsigned long area = height * width;

    // **************************
    // * DECLARATION, ALOCATION *
    // **************************

    *bboxes_p = malloc(4 * area / 2 * sizeof(**bboxes_p));  // bigger than necessary
    if (*bboxes_p == NULL) {
        fprintf(stderr, "failed to allocate the full bboxes array\n");
        return 1;
    }
    clusters = calloc(area, sizeof(*clusters));  // 2d image of the clusters
    if (clusters == NULL) {  // even if all elements are not zero, calloc is faster than malloc + set 0
        free(*bboxes_p);
        fprintf(stderr, "failed to allocate the clusters array\n");
        return 1;
    }
    merge = malloc(area / 2 * sizeof(*merge));  // like hash table
    if (merge == NULL) {
        free(*bboxes_p);
        free(clusters);
        fprintf(stderr, "failed to allocate an internal array\n");
        return 1;
    }

    // *****************
    // * FIND CLUSTERS *
    // *****************

    for (shift_img = 0; shift_img < area; ++shift_img) {
        if (!binary[shift_img]) continue;  // case black pixel
        // ldiv_t ij = ldiv(shift_img, width);
        // i = ij.quot, j = ij.rem;  // same as i = shift_img / width, j = shift_img % width;
        i = shift_img / width, j = shift_img % width;
        clus_top = i ? clusters[shift_img - width] : 0;  // test for 0 padding
        clus_left = j ? clusters[shift_img - 1] : 0;  // test for 0 padding
        if (!clus_top && !clus_left) {  // case new cluster
            stride = 4 * curr_clus;
            (*bboxes_p)[stride] = (npy_int16)i, (*bboxes_p)[stride + 1] = (npy_int16)j; // anchor
            (*bboxes_p)[stride + 2] = 1, (*bboxes_p)[stride + 3] = 1;  // shape
            merge[curr_clus] = 0;  // no merge
            clusters[shift_img] = ++curr_clus;  // next cluster
        } else if (clus_top == clus_left) {  // case same clusters
            clusters[shift_img] = clus_top;
        } else if (!clus_top) {  // case same as left
            stride = 4 * (clus_left - 1) + 1;
            clus_top = (unsigned long)((*bboxes_p)[stride] + (*bboxes_p)[stride + 2]);  // right pos
            clus_top = j + 1 > clus_top ? j + 1 : clus_top;  // new right pos + 1
            (*bboxes_p)[stride + 2] = (npy_int16)clus_top - (*bboxes_p)[stride];  // width
            clusters[shift_img] = clus_left;
        } else if (!clus_left) {  // case same as top
            stride = 4 * (clus_top - 1);
            clus_left = (unsigned long)((*bboxes_p)[stride] + (*bboxes_p)[stride + 2]);  // bottom pos
            clus_left = i + 1 > clus_left ? i + 1 : clus_left;  // new bottom pos + 1
            (*bboxes_p)[stride + 2] = (npy_int16)clus_left - (*bboxes_p)[stride];  // hight
            clusters[shift_img] = clus_top;
        } else {  // case merge
            if (clus_top > clus_left) {  // guaranteed that clu > merge[clu-1]
                stride = clus_left;
                clus_left = clus_top, clus_top = stride;
            }
            clus_top = merge[clus_top-1] ? (unsigned long)merge[clus_top-1] : clus_top;
            merge[clus_left-1] = (npy_int16)clus_top;
            clusters[shift_img] = clus_top;
            MergeBBoxes((*bboxes_p) + 4 * (clus_left-1), (*bboxes_p) + 4 * (clus_top-1));
        }
    }

    // ******************
    // * MERGE CLUSTERS *
    // ******************

    for (unsigned long clu_m1 = 0; clu_m1 < curr_clus; ++clu_m1) {  // merge
        if (merge[clu_m1]) { // we have always clu > merge[clu]
            MergeBBoxes((*bboxes_p) + 4 * clu_m1, (*bboxes_p) + 4 * (merge[clu_m1]-1));
        }
    }
    for (unsigned long clu_m1 = 0; clu_m1 < curr_clus; ++clu_m1) {  // remove too tall bboxes
        stride = 4 * clu_m1;
        if ((*bboxes_p)[stride + 2] > max_size || (*bboxes_p)[stride + 3] > max_size) {
            merge[clu_m1] = 0;  // hack to skip that one
        }
    }
    *n = 0;
    for (npy_intp clu_m1 = 0; clu_m1 < (npy_intp)curr_clus; ++clu_m1) {  // concatenate, skip some bboxes
        if (!merge[clu_m1]) {
            if (*n != clu_m1) {
                memcpy((*bboxes_p) + 4 * (*n), (*bboxes_p) + 4 * clu_m1, 4 * sizeof(**bboxes_p));
            }
            ++(*n);
        }
    }

    fprintf(stderr, "*n=%ld, curr_clus=%ld, area/2=%ld\n", *n, curr_clus, area/2);

    free(clusters);
    free(merge);
    *bboxes_p = realloc(*bboxes_p, 4 * (unsigned long)(*n) * sizeof(**bboxes_p));
    if (*bboxes_p == NULL && *n) {
        fprintf(stderr, "failed to free the piece at the end of full_bboxes\n");
        return 1;
    }

    return 0;
}


static PyObject* FindBBoxes(PyObject* self, PyObject* args) {
    // Find the bboxes
    npy_int16* full_bboxes;
    PyArrayObject *binary;
    PyObject *bboxes;
    long max_size;
    long unsigned height, width;
    npy_intp bboxes_shape[2];
    int error;

    if (!PyArg_ParseTuple(args, "O!l", &PyArray_Type, &binary, &max_size)) {
        return NULL;
    }
    if (CheckImg(binary, NPY_UINT8)) {
        return NULL;
    }
    if (max_size < 1){
        PyErr_SetString(PyExc_ValueError, "'max_size' has to be >= 1");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    height = (unsigned long)PyArray_DIM(binary, 0), width = (unsigned long)PyArray_DIM(binary, 1);
    error = FindClustersAndBBoxes(
        &full_bboxes,
        bboxes_shape,
        (unsigned char *)PyArray_DATA(binary),
        height,
        width,
        (npy_int16)max_size
    );
    Py_END_ALLOW_THREADS
    if (error) {
        return PyErr_NoMemory();
    }

    // bboxes_shape[1] = 4;
    // bboxes = PyArray_EMPTY(2, bboxes_shape, NPY_INT16, 0);  // c contiguous
    // if (bboxes == NULL) {
    //     free(full_bboxes);
    //     return PyErr_NoMemory();
    // }
    // Py_BEGIN_ALLOW_THREADS
    // memcpy(PyArray_DATA(bboxes), full_bboxes, bboxes_shape[0] * bboxes_shape[1] * sizeof(*full_bboxes));
    // free(full_bboxes);
    // Py_END_ALLOW_THREADS

    bboxes_shape[1] = 4;
    bboxes = PyArray_SimpleNewFromData(2, bboxes_shape, NPY_INT16, full_bboxes);
    if (bboxes == NULL) {
        free(full_bboxes);
        return PyErr_NoMemory();
    }

    return bboxes;
}


static PyMethodDef findBBoxesMethods[] = {
    {"find_bboxes", FindBBoxes, METH_VARARGS, "Find the bboxes of the binary image."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_find_bboxes = {
    PyModuleDef_HEAD_INIT,
    "find_bboxes",
    "Fast Binary Clustering.",
    -1,
    findBBoxesMethods
};


PyMODINIT_FUNC PyInit_c_find_bboxes(void) {
    import_array();
    if ( PyErr_Occurred() ) {
        return NULL;
    }
    return PyModule_Create(&c_find_bboxes);
}
