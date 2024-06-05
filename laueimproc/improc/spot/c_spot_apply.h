#define PY_SSIZE_T_CLEAN
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdio.h>


int ApplyToRois(
    PyArrayObject* out,
    PyByteArrayObject* data,
    PyArrayObject* bboxes,
    int (*func)(PyArrayObject* out, const npy_intp b, const npy_float32* roi, const npy_int16 bbox[4])
) {
    /* Apply the function on each roi.

    Parameters
    ----------
    out : np.ndarray
        A c contiguous array of shape (n, ...) with n the number of spots.
        The result is writen inplace inside this array.
        Shape is assumed to be already checked
    data : bytearray
        The rois flatten content. The len is check in this function.
    bboxes : np.ndarray
        The int16 array of shape(n, 4), with n the number of spots.
        Shape and dtype is assumed to be already check.
    func : callable
        The function pointer to apply to each roi.
    */
    npy_float32* rawdata;
    npy_intp datalen = (npy_intp)PyByteArray_Size((PyObject *)data), shift = 0, area;
    const npy_intp n = PyArray_DIM(bboxes, 0);
    npy_int16 bbox[4];

    if (datalen % sizeof(npy_float)) {
        fprintf(stderr, "data length is not a multiple of float32 length\n");
        return 1;
    }
    datalen /= sizeof(npy_float);
    rawdata = (npy_float32 *)PyByteArray_AsString((PyObject *)data);
    if (rawdata == NULL) {
        fprintf(stderr, "data is empty\n");
        return 1;
    }

    for (npy_intp b = 0; b < n; ++b) {
        bbox[0] = *(npy_int16 *)PyArray_GETPTR2(bboxes, b, 0);
        bbox[1] = *(npy_int16 *)PyArray_GETPTR2(bboxes, b, 1);
        bbox[2] = *(npy_int16 *)PyArray_GETPTR2(bboxes, b, 2);
        bbox[3] = *(npy_int16 *)PyArray_GETPTR2(bboxes, b, 3);
        area = (npy_intp)(bbox[2]) * (npy_intp)(bbox[3]);
        if (!area) {
            fprintf(stderr, "the bbox %ld has zero area\n", b);
            return 1;
        }
        if (shift + area > datalen) {
            fprintf(stderr, "the data length 4*%ld is too short to fill roi of index %ld\n", datalen, b);
            return 1;
        }
        if ((*func)(out, b, rawdata+shift, bbox)) {
            return 1;
        }
        shift += area;
    }
    if (datalen != shift) {  // data to long
        fprintf(stderr, "the data length 4*%ld is too long to fill a total area of %ld pxls.\n", datalen, shift);
        return 1;
    }
    return 0;
}
