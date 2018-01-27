#include <Python.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

#define color(row, col) ((((row) & 1) << 1) + ((col) & 1))

static PyObject *get_frame_data(PyObject *self, PyObject *args)
{
    int width, height, offset, index;
    char *filename;
    PyObject *data;
    FILE *src;
    int r_off, c_off;
    unsigned short *line;

    src = NULL;
    line = NULL;

    if (!PyArg_ParseTuple(args, "Osiiii", &data, &filename,
                          &index, &width, &height, &offset)) {
        goto err;
    }

    src = fopen(filename, "rb");
    if (!src) {
        goto err;
    }

    switch (index) {
    case 0:
        r_off = c_off = 1;
        break;
    case 1:
        r_off = 0;
        c_off = 1;
        break;
    case 2:
        r_off = c_off = 0;
        break;
    case 3:
        r_off = 1;
        c_off = 0;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "index must be between 0 and 3");
        goto err;
    }

    if (fseek(src, offset, SEEK_SET) != 0) {
        PyErr_SetFromErrno(PyExc_IOError);
        goto err;
    }

    line = (unsigned short *)malloc(width * sizeof(unsigned short));
    if (!line) {
        PyErr_NoMemory();
        goto err;
    }
    
    for (int row = 0; row < height; ++row) {
        if (fread(line, 2, width, src) != width) {
            PyErr_SetString(PyExc_IOError, "fread faliure");
            goto err;
        }
        for (int col = 0; col < width; ++col) {
            int rr = row + r_off - 1;
            if (rr >= 0) {
                int cc = col + c_off - 1;
                if (cc >= 0) {
                    int c = color(row, col);
                    unsigned short *out =
                        (unsigned short *)PyArray_GETPTR3(data, rr, cc, c);
                    *out = line[col];
                }
            }
        }
    }

    free(line);
    fclose(src);
    
    Py_INCREF(Py_None);
    return Py_None;

  err:
    if (line) {
        free(line);
    }
    if (src) {
        fclose(src);
    }
    return NULL;
}


static PyMethodDef _makearq_methods[] = {
    {"get_frame_data", (PyCFunction) get_frame_data, METH_VARARGS,
     "Get the data stored in one ARW frame"},
    {NULL, NULL, 0, NULL}
};


void init_makearq(void)
{
    Py_InitModule("_makearq", _makearq_methods);
}
