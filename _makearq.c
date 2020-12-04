/*
 *   make_arq - A tool for generating Sony A7RIII Pixel-Shift ARQ files
 *   Copyright (C) 2018 Alberto Griggio <alberto.griggio@gmail.com>
 *
 *   make_arq is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   make_arq is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *   
 */
#include <Python.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

#define color(row, col) ((((row) & 1) << 1) + ((col) & 1))
#define dngcolor(row, col) (0x94949494 >> ((((row) << 1 & 14) + ((col) & 1)) << 1) & 3)

static PyObject *get_sony_frame_data(PyObject *self, PyObject *args)
{
    int width, height, offset;
    int factor, rowstart, colstart;
    char *filename;
    PyObject *data;
    FILE *src;
    int r_off, c_off;
    int is_dng;
    unsigned short *line;

    src = NULL;
    line = NULL;

    if (!PyArg_ParseTuple(args, "Osiiiiiiiii", &data, &filename,
                          &width, &height, &offset, &factor,
                          &r_off, &c_off, &rowstart, &colstart, &is_dng)) {
        goto err;
    }

    src = fopen(filename, "rb");
    if (!src) {
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
            int rr = (row + r_off) * factor + rowstart;
            if (rr >= 0 && rr < height * factor) {
                int cc = (col + c_off) * factor + colstart;
                if (cc >= 0 && cc < width * factor) {
                    int c = is_dng ? dngcolor(row, col) : color(row, col);
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


static PyObject *get_fuji_frame_data(PyObject *self, PyObject *args)
{
    int factor, rowstart, colstart;
    PyObject *data, *im;
    int r_off, c_off;
    int width, height;
    int is_dng;

    if (!PyArg_ParseTuple(args, "OiiiOiiiii", &im, &height, &width, &factor,
                          &data, &r_off, &c_off, &rowstart, &colstart,
                          &is_dng)) {
        goto err;
    }

    for (int y = 0; y < height; ++y) {
        int rr = (y + r_off) * factor + rowstart;
        if (rr >= 0 && rr < height * factor) {
            for (int x = 0; x < width; ++x) {
                int cc = (x + c_off) * factor + colstart;
                if (cc >= 0 && cc < width * factor) {
                    unsigned short *v =
                        (unsigned short *)PyArray_GETPTR2(im, y, x);
                    int c = is_dng ? dngcolor(y, x) : color(y, x);
                    unsigned short *out =
                        (unsigned short *)PyArray_GETPTR3(data, rr, cc, c);
                    *out = *v;
                }
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;

  err:
    return NULL;
}


static PyMethodDef _makearq_methods[] = {
    {"get_sony_frame_data", (PyCFunction) get_sony_frame_data, METH_VARARGS,
     "Get the data stored in one ARW frame"},
    {"get_fuji_frame_data", (PyCFunction) get_fuji_frame_data, METH_VARARGS,
     "Get the data stored in one RAF frame"},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION == 2

void init_makearq(void)
{
    Py_InitModule("_makearq", _makearq_methods);
}

#else

static struct PyModuleDef _makearqmodule = {
    PyModuleDef_HEAD_INIT,
    "_makearq",
    NULL,
    -1,
    _makearq_methods
};

PyMODINIT_FUNC
PyInit__makearq(void)
{
    return PyModule_Create(&_makearqmodule);
}

#endif
