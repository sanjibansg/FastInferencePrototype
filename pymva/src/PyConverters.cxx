// @(#)root/tmva/pymva $Id$
// Author: Sanjiban Sengupta, 2021

#include <Python.h>
#include "TMVA/PyConverters.h"



#include <memory>
#include <cctype>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


#include "TMVA/Types.h"
#include "Rtypes.h"
#include "TString.h"
#include "TMVA/MsgLogger.h"


#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"



using namespace TMVA;
using namespace TMVA::Experimental;
using namespace TMVA::Experimental::SOFIE;


namespace TMVA{

// Declaring Global variables
PyObject *fModuleBuiltin = NULL;
PyObject *fEval = NULL;
PyObject *fOpen = NULL;

PyObject *fModulePickle = NULL;
PyObject *fPickleDumps = NULL;
PyObject *fPickleLoads = NULL;

PyObject *fMain = NULL;
PyObject *fGlobalNS = NULL;
PyObject *fPyReturn = NULL;

MsgLogger* fLogger;
MsgLogger& Log()  { return *fLogger; }


///////////////////////////////////////////////////////////////////////////////
/// Check Python interpreter initialization status
///
/// \return Boolean whether interpreter is initialized

int PyIsInitialized()
{
   if (!Py_IsInitialized()) return kFALSE;
   if (!fEval) return kFALSE;
   if (!fModuleBuiltin) return kFALSE;
   if (!fPickleDumps) return kFALSE;
   if (!fPickleLoads) return kFALSE;
   return kTRUE;
}


void PyInitialize()
{
   TMVA::MsgLogger Log;

   bool pyIsInitialized = PyIsInitialized();
   if (!pyIsInitialized) {
      Py_Initialize();
   }

   if (!pyIsInitialized) {
      _import_array();
   }

   fMain = PyImport_AddModule("__main__");
   if (!fMain) {
      Log << kFATAL << "Can't import __main__" << Endl;
      Log << Endl;
   }

   fGlobalNS = PyModule_GetDict(fMain);
   if (!fGlobalNS) {
      Log << kFATAL << "Can't init global namespace" << Endl;
      Log << Endl;
   }

   #if PY_MAJOR_VERSION < 3
   //preparing objects for eval
   PyObject *bName =  PyUnicode_FromString("__builtin__");
   // Import the file as a Python module.
   fModuleBuiltin = PyImport_Import(bName);
   if (!fModuleBuiltin) {
      Log << kFATAL << "Can't import __builtin__" << Endl;
      Log << Endl;
   }
   #else
   //preparing objects for eval
   PyObject *bName =  PyUnicode_FromString("builtins");
   // Import the file as a Python module.
   fModuleBuiltin = PyImport_Import(bName);
   if (!fModuleBuiltin) {
      Log << kFATAL << "Can't import builtins" << Endl;
      Log << Endl;
   }
   #endif

   PyObject *mDict = PyModule_GetDict(fModuleBuiltin);
   fEval = PyDict_GetItemString(mDict, "eval");
   fOpen = PyDict_GetItemString(mDict, "open");

   Py_DECREF(bName);
   Py_DECREF(mDict);
   //preparing objects for pickle
   PyObject *pName = PyUnicode_FromString("pickle");
   // Import the file as a Python module.
   fModulePickle = PyImport_Import(pName);
   if (!fModulePickle) {
      Log << kFATAL << "Can't import pickle" << Endl;
      Log << Endl;
   }
   PyObject *pDict = PyModule_GetDict(fModulePickle);
   fPickleDumps = PyDict_GetItemString(pDict, "dump");
   fPickleLoads = PyDict_GetItemString(pDict, "load");

   Py_DECREF(pName);
   Py_DECREF(pDict);
}



PyObject *Eval(TString code,PyObject *fLocalNS)
{
   if(!PyIsInitialized()) PyInitialize();
   PyObject *pycode = Py_BuildValue("(sOO)", code.Data(), fGlobalNS, fLocalNS);
   PyObject *result = PyObject_CallObject(fEval, pycode);
   Py_DECREF(pycode);
   return result;
}



///////////////////////////////////////////////////////////////////////////////
// Finalize Python interpreter

void PyFinalize()
{
   Py_Finalize();
   if (fEval) Py_DECREF(fEval);
   if (fModuleBuiltin) Py_DECREF(fModuleBuiltin);
   if (fPickleDumps) Py_DECREF(fPickleDumps);
   if (fPickleLoads) Py_DECREF(fPickleLoads);
   if(fMain) Py_DECREF(fMain);//objects fGlobalNS and fLocalNS will be free here
}


///////////////////////////////////////////////////////////////////////////////
/// Set program name for Python interpeter
///
/// \param[in] name Program name

void PySetProgramName(TString name)
{
   #if PY_MAJOR_VERSION < 3
   Py_SetProgramName(const_cast<char*>(name.Data()));
   #else
   Py_SetProgramName((wchar_t *)name.Data());
   #endif
}


///////////////////////////////////////////////////////////////////////////////

size_t mystrlen(const char* s) { return strlen(s); }

///////////////////////////////////////////////////////////////////////////////

size_t mystrlen(const wchar_t* s) { return wcslen(s); }

///////////////////////////////////////////////////////////////////////////////
/// Get program name from Python interpreter
///
/// \return Program name

TString Py_GetProgramName()
{
   auto progName = ::Py_GetProgramName();
   return std::string(progName, progName + mystrlen(progName));
}


///////////////////////////////////////////////////////////////////////////////
/// Serialize Python object
///
/// \param[in] path Path where object is written to file
/// \param[in] obj Python object
///
/// The input Python object is serialized and written to a file. The Python
/// module `pickle` is used to do so.

void Serialize(TString path, PyObject *obj)
{
   if(!PyIsInitialized()) PyInitialize();

   PyObject *file_arg = Py_BuildValue("(ss)", path.Data(),"wb");
   PyObject *file = PyObject_CallObject(fOpen,file_arg);
   PyObject *model_arg = Py_BuildValue("(OO)", obj,file);
   PyObject *model_data = PyObject_CallObject(fPickleDumps , model_arg);

   Py_DECREF(file_arg);
   Py_DECREF(file);
   Py_DECREF(model_arg);
   Py_DECREF(model_data);
}

///////////////////////////////////////////////////////////////////////////////
/// Unserialize Python object
///
/// \param[in] path Path to serialized Python object
/// \param[in] obj Python object where the unserialized Python object is loaded
///  \return Error code

Int_t UnSerialize(TString path, PyObject **obj)
{
   // Load file
   PyObject *file_arg = Py_BuildValue("(ss)", path.Data(),"rb");
   PyObject *file = PyObject_CallObject(fOpen,file_arg);
   if(!file) return 1;

   // Load object from file using pickle
   PyObject *model_arg = Py_BuildValue("(O)", file);
   *obj = PyObject_CallObject(fPickleLoads , model_arg);
   if(!obj) return 2;

   Py_DECREF(file_arg);
   Py_DECREF(file);
   Py_DECREF(model_arg);

   return 0;
}

///////////////////////////////////////////////////////////////////////////////
/// Execute Python code from string
///
/// \param[in] code Python code as string
/// \param[in] errorMessage Error message which shall be shown if the execution fails
/// \param[in] start Start symbol
///
/// Helper function to run python code from string in local namespace with
/// error handling
/// `start` defines the start symbol defined in PyRun_String (Py_eval_input,
/// Py_single_input, Py_file_input)

void PyRunString(TString code, PyObject *fLocalNS, TString errorMessage="Failed to run python code", int start=Py_single_input) {
   fPyReturn = PyRun_String(code, start, fGlobalNS, fLocalNS);
   if (!fPyReturn) {
      Log() << kWARNING << "Failed to run python code: " << code << Endl;
      Log() << kWARNING << "Python error message:" << Endl;
      PyErr_Print();
      Log() << kFATAL << errorMessage << Endl;
   }
}


namespace PyKeras {
void ConvertToRoot(TString filepath){

 if (!PyIsInitialized()) {
      PyInitialize();
   }

    // Set up private local namespace for each method instance
   PyObject *fLocalNS = PyDict_New();
   if (!fLocalNS) {
      Log() << kFATAL << "Can't init local namespace" << Endl;
   }

   PyRunString("from keras.models import load_model",fLocalNS);
   PyRunString(TString::Format("model=load_model(%s)",filepath),fLocalNS);
   PyRunString(TString::Format("model.load_weights(%s)",filepath),fLocalNS);
   PyRunString("modelProp=[]",fLocalNS);
   PyRunString("for idx in range(len(model.layers)):\n"
            "	layerProp=[]"
            "	layerProp.extend([x for x in [getattr(obj,'__name__',None) for obj in [getattr(model.get_layer(index=idx),i,None) for i in ['__class__','__activation__']]]])"
            "	layerProp.extend([x for x in [getattr(obj,'name',None) for obj in [getattr(model.get_layer(index=idx),i,None) for i in ['input','output']]]])"
            "	layerProp.extend([x for x in [getattr(model.get_layer(index=idx),i,None) for i in ['input_shape','output_shape'] ]])"
            "	modelProp.append(layerProp)",fLocalNS);



}
   }

namespace PyTorch {
void ConvertToRoot(TString filepath){
std::cout<<filepath;

}

}




  }

