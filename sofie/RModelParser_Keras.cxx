// @(#)root/tmva/pymva $Id$
// Author: Sanjiban Sengupta, 2021

#include <Python.h>
#include "TMVA/PyInitialize.hxx"
#include "TMVA/RModelParser_Keras.hxx"



#include <memory>
#include <cctype>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


#include "TMVA/Types.h"
#include "TMVA/SOFIE_common.hxx"
#include "Rtypes.h"
#include "TString.h"
#include "TMVA/MsgLogger.h"


#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"



using namespace TMVA;
using namespace TMVA::Experimental;
using namespace TMVA::Experimental::SOFIE;





namespace TMVA{
namespace Experimental{

PyObject *fGlobalNS = NULL;
PyObject *fPyReturn = NULL;

MsgLogger* fLogger;
MsgLogger& Log()  { return *fLogger; }

unordered_map<std::string, int> Type =
    {
        {"dense", LayerType::DENSE},
        {"relu", LayerType::RELU},
        {"permute", LayerType::TRANSPOSE}
    };

unordered_map<std::string,int>dType=
{
      {"float32", ETensorType::FLOAT}
};

namespace INTERNAL{




   std::unique_ptr<ROperator> make_ROperator_Gemm(std::string input,std::string output,std::string kernel,std::string bias,std::string input_type){

   std::unique_ptr<ROperator> op;

   float attr_alpha =1.0;
   float attr_beta =1.0;
   int_t attr_transA =0;
   int_t attr_transB =1;

   switch(input_type){
   case ETensorType::FLOAT:
         op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, input, kernel, bias, output);
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + input_type);
   }

   return std::move(op);


   }

   std::unique_ptr<ROperator> make_ROperator_Relu(std::string input, std::string output, std::string input_type){
      std::unique_ptr<ROperator> op;
      switch(input_type){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Relu<float>(input, output));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + input_type);
         }
   return std::move(op);
   }

   std::unique_ptr<ROperator> make_ROperator_Transpose(std::string input, std::string output, std::vector<int_t> dims, std::string input_type){

   std::unique_ptr<ROperator> op;
   std::vector<int_t> attr_perm=dims;

   switch(input_type){
   case ETensorType::FLOAT:
      if (!attr_perm.empty()){
         op.reset(new ROperator_Transpose<float>(attr_perm, input, output);
      }else{
         op.reset(new ROperator_Transpose<float> (input, output);
      }
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + input_type);
   }

   return std::move(op);

   }



}





namespace PyKeras {
RModel Parse(std::string filepath){

   char sep = '/';
   #ifdef _WIN32
   sep = '\\';
   #endif

   size_t i = filename.rfind(sep, filename.length());
   if (i != std::string::npos){
      filename = (filename.substr(i+1, filename.length() - i));
   }

   std::time_t ttime = std::time(0);
   std::tm* gmt_time = std::gmtime(&ttime);
   std::string parsetime (std::asctime(gmt_time));

   RModel rmodel(filename, parsetime);

   if (!PyIsInitialized()) {
      PyInitialize();
   }

    // Set up private local namespace for each method instance
   PyObject *fLocalNS = PyDict_New();
   if (!fLocalNS) {
      Log() << kFATAL << "Can't init local namespace" << Endl;
   }


   PyRunString("from keras.models import load_model",fLocalNS);
   PyRunString(TString::Format("model=load_model('%s')",filepath.c_str()),fLocalNS);
   PyRunString(TString::Format("model.load_weights('%s')",filepath.c_str()),fLocalNS);
   PyRunString("modelData=[]",fLocalNS);
   PyRunString("for idx in range(len(model.layers)):\n"
            "	layerData={}\n"
            "  layerData.update({(k,v) for (k,v) in {key:getattr(value,'__name__',None) for (key,value)  in {i:getattr(model.get_layer(index=idx),i,None) for i in ['__class__','activation']}.items()}.items()})\n"
            "	layerData.update({(k,v) for (k,v) in {i:getattr(model.get_layer(index=idx),i,None) for i in ['name','dtype','input_shape','output_shape','dims']}.items()})\n"
            "	layerData.update({(k,v) for (k,v) in {key:getattr(value,'name',None) for (key,value)  in {i:getattr(model.get_layer(index=idx),i,None) for i in ['input','output','kernel','bias']}.items()}.items()})\n"
            "	modelProp.append(layerData)",fLocalNS);


   Py_ssize_t modelIterator, modelSize;
   PyObject* pModel = PyDict_GetItemString(fLocalNS,"modelProp");
   PyObject* layers;
   modelSize = PyList_Size(pModel);

   for(modelIterator=0;modelIterator<modelSize;++modelIterator){
      layer=PyList_GetItem(pModel,modelIterator);

      std::string type(PyString_AsString(PyDict_GetItemString(layer,"__class__")));
      std::string name(PyString_AsString(PyDict_GetItemString(layer,"name")));
      std::string activation(PyString_AsString(PyDict_GetItemString(layer,"activation")));
      std::string dtype(PyString_AsString(PyDict_GetItemString(layer,"dtype")));
      std::string input(PyString_AsString(PyDict_GetItemString(layer,"input")));
      std::string output(PyString_AsString(PyDict_GetItemString(layer,"output")));
      std::string output(PyString_AsString(PyDict_GetItemString(layer,"kernel")));
      std::string output(PyString_AsString(PyDict_GetItemString(layer,"bias")));

      PyObject* input_shape  = PyDict_GetItemString(layer,"input_shape");
      PyObject* output_shape = PyDict_GetItemString(layer,"output_shape");


      if(dType.find(dtype)==dType.end())
         throw std::runtime_error("Type error: Layer data type not yet registered in TMVA SOFIE");



      switch(Type.find(toLower(layerType))->second){
         case LayerType::DENSE : {
            switch(dType.find(dtype)->second){

               case ETensorType::FLOAT :{
                  if(activation != "linear"){
                     rmodel.AddOperator(std::move(INTERNAL::makeROperator_Gemm(input,name+"_gemm",kernel,shape,input_type)));

                     switch(Type.find(toLower(activation))->second){
                        case LayerType::RELU: {
                           rmodel.AddOperator(std::move(INTERNAL::makeROperator_Relu(name+"_gemm",output,kernel,shape,dtype)));
                           break;
                        default: throw std::runtime_error("Activation error: TMVA SOFIE does not yet suppport Activation type"+activation);
                        }
                        }
                        }
                  else
                     rmodel.AddOperator(std::move(INTERNAL::makeROperator_Gemm(input,output,kernel,shape,dtype)));
                     break;
                     }
               default: throw std::runtime_error("Type error: TMVA SOFIE does not yet suppport layer data type"+dtype);
               }
               }

         case LayerType::RELU: {
            rmodel.AddOperator(std::move(INTERNAL::makeROperator_Relu(input,output,dtype)));  break;
            }
         case LayerType::TRANSPOSE: {
            PyObject* permute=PyDict_GetItemString(layer,"dims");
            std::vector<int_t>dims;
            for(Py_ssize_t tupleIter=0;tupleIter<PyTuple_Size(permute);++tupleIter)
               dims.push_back(PyTuple_GetItem(permute,tupleIter));
            rmodel.AddOperator(std::move(INTERNAL::makeROperator_Transpose(input,output,dims,dtype))); break;
            }
         default: throw std::runtime_error("Layer error: TMVA SOFIE does not yet suppport layer type"+dtype);
         }

         }


   //model.get_weights() returns weights in list of numpy array format
   PyRunString("weight=[]",fLocalNS);
   PyRunString("for idx in range(len(model.get_weights())):\n"
               "  weightProp={}\n"
               "  weightProp['name']=model.weights[idx].name\n"
               "  weightProp['dtype']=(model.get_weights())[idx].dtype.name\n"
               "  weightProp['value']=(model.get_weights())[idx]\n"
               "  weight.append(weightProp)",fLocalNS);

   PyObject *item,*tensorName,weightType;
   std::vector<RTensor<float>>weights;

   for (Py_ssize_t it = 0; it < PyList_Size(pWeights); it++) {
   item  = PyList_GetItem(pWeights, it);
   tensorName  = PyList_GetItem(item,0);
   std::string weightDtype(PyString_AsString(PyList_GetItem(item,1)));

   if(dType.find(weightType)==dType.end())
      throw std::runtime_error("Type error: Initialized tensor type not yet registered in TMVA SOFIE");

   switch(dType.find(type)->second){
       case ETensorType::FLOAT :
       array = PyList_Getitem(item,2);
       RTensor value = getArray(array);
       std::shared_ptr<void> data(malloc(value.GetSize() * sizeof(float)), free);
       std::memcpy(data.get(), value.GetData();, value.GetSize() * sizeof(float));
       rmodel.AddInitializedTensor(PyString_AsString(tensorName), ETensorType::FLOAT, value.GetShape(), data);
       break;

       default: throw std::runtime_error("Type error: TMVA SOFIE does not yet suppport layer type"+dtype);
      }
     }


      PyRunString("inputs=model.input_names\n",fLocalNS);
      PyRunString("inputShapes=model.input_shape\n",fLocalNS);
      PyObject* pInputs      = PyDict_GetItemString(fLocalNS,"inputs");
      PyObject* pInputShapes = PyDict_GetItemString(fLocalNS,"inputShapes");
      PyObject* pInputTypes  = PyDict_GetItemString(fLocalNS,"inputTypes");
      for(Py_ssize_t inputIter = 0; inputIter < PyList_Size(pInputs);++inputIter){

         std::string inputDType(PyString_AsString(PyList_GetItem(pInputTypes,i)));
         if(dType.find(inputDType)==dType.end())
            throw std::runtime_error("Type error: Initialized tensor type not yet registered in TMVA SOFIE");

      switch(dType.find(inputDType)->second){

         case 1:
         std::vector<int>shape;
         std::string name(PyString_AsString(PyList_GetItem(pInputs,i)));

         PyObject* shapeTuple=PyList_GetItem(pInputShapes,i);
         std::string inputType(PyString_AsString(PyList_GetItem(pInputs,i)));
         for(Py_ssize_t tupleIter=1;tupleIter<PyTuple_Size(shapeTuple);++tupleIter){
               shape.push_back((int)PyInt_AsLong(PyTuple_GetItem(shapeTuple,tupleiter)));
         }

         rmodel.AddInputTensorInfo(name, type, fShape);
         break;

         default: throw std::runtime_error("Type error: TMVA SOFIE does not yet suppport layer type"+dType.find(weightType)->first);


      }




      }












     }
   }
}
