#include <Python.h>
int main()
{
    PyObject *strret, *mymod, *strfunc, *strargs;
    char *cstrret;
    Py_Initialize();
    mymod = PyImport_ImportModule("reverse");
    strfunc = PyObject_GetAttrString(mymod, "rstring");
    char a[3];
    a[0]='a';
    a[1]='b';
    a[2]='c';
    strargs = Py_BuildValue("(s)", a);
    strret = PyEval_CallObject(strfunc, strargs);
    PyArg_Parse(strret, "s", &cstrret);
    printf("Reversed string: %s\n", cstrret);
    Py_Finalize();
    return 0;

    }
