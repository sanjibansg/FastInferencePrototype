#define PY_SSIZE_T_CLEAN
#include <Python.h>

int
main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);  /* optional but recommended */
    Py_Initialize();
    PyRun_SimpleString("from time import time,ctime\n"
                       "print('Today is', ctime(time()))\n");
    PyObject* fLocalNS;
    PyObject* fGlobalNS;
    PyObject* fMain;
    fMain = PyImport_AddModule("__main__");
    fGlobalNS = PyModule_GetDict(fMain);

    fLocalNS=PyDict_New();                   
    PyRun_String("from keras.models import load_model",Py_single_input,fGlobalNS,fLocalNS);
     PyRun_String("model=load_model('model1.h5')",Py_single_input,fGlobalNS,fLocalNS);  
       
    PyRun_String("model.load_weights('model1.h5')",Py_single_input,fGlobalNS,fLocalNS);      
    PyRun_String("weights=model.get_weights()",Py_single_input,fGlobalNS,fLocalNS); 
    
    
    
       PyObject* pWeights = PyDict_GetItemString(fLocalNS, "weights");

    
    
                                                     
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    PyMem_RawFree(program);
    return 0;
}
