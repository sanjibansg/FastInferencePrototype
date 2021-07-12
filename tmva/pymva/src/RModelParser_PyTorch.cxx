
#include "TMVA/RModelParser_PyTorch.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

std::unordered_map<std::string, NodeType> NType =
    {
        {"'onnx::gemm'", NodeType::GEMM},
        {"'onnx::relu'", NodeType::RELU},
        {"'onnx::transpose'", NodeType::TRANSPOSE}
    };




namespace PyTorch{

RModel Parse(std::string filename, std::vector<std::vector<size_t>> inputShapes, ETensorType dtype){
    char sep='/';
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


    Py_Initialize();
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* fGlobalNS = PyModule_GetDict(main);
    PyObject* fLocalNS = PyDict_New();
    if (!fGlobalNS) {
       throw std::runtime_error("Can't init global namespace for Python");
       }
    if (!fLocalNS) {
       throw std::runtime_error("Can't init local namespace for Python");
       }



    switch(dtype){

        case ETensorType::FLOAT : {

            //Extracting model information
            //Model is converted to ONNX graph format
            //using PyTorch's internal function with the input shape provided
            PyRunString("globals().update(locals())",fGlobalNS,fLocalNS);
            PyRunString("import torch",fGlobalNS,fLocalNS);
            PyRunString("import torch",fGlobalNS,fGlobalNS);
            PyRunString("from torch.onnx.utils import _model_to_graph",fGlobalNS,fLocalNS);
            PyRunString(TString::Format("model= torch.jit.load('%s')",filename.c_str()),fGlobalNS,fLocalNS);
            PyRunString("globals().update(locals())",fGlobalNS,fLocalNS);
            PyRunString("model.float()",fGlobalNS,fLocalNS);
            PyRunString("model.eval()",fGlobalNS,fLocalNS);

            //Building dummy inputs for the model
            PyRunString("dummy=[]",fGlobalNS,fLocalNS);
            for(auto it=0;it<inputShapes.size();++it){
                PyRunString("inputShape=[]",fGlobalNS,fLocalNS);
                for(auto itr=0;itr<inputShapes[it].size();++itr){
                    PyRunString(TString::Format("inputShape.append(%d)",(int)inputShapes[it][itr]),fGlobalNS,fLocalNS);
                }
                PyRunString("dummy.append(torch.rand(*inputShape))",fGlobalNS,fLocalNS);
            }

            //Finding example outputs from dummy
            PyRunString("output=model(*dummy)",fGlobalNS,fLocalNS);

            //Getting the ONNX graph from model using the dummy inputs and example outputs
            PyRunString("graph=_model_to_graph(model,dummy,example_outputs=output)",fGlobalNS,fLocalNS);



            //Extracting the model information in lost modelData
            PyRunString("modelData=[]",fGlobalNS,fLocalNS);
            PyRunString("for i in graph[0].nodes():\n"
                        "    globals().update(locals())\n"
                        "    nodeData=[]\n"
                        "    nodeData.append(i.kind())\n"
                        "    nodeAttributeNames=[x for x in i.attributeNames()]\n"
                        "    nodeAttributes={j:i[j] for j in nodeAttributeNames}\n"
                        "    nodeData.append(nodeAttributes)\n"
                        "    nodeInputs=[x for x in i.inputs()]\n"
                        "    nodeInputNames=[x.debugName() for x in nodeInputs]\n"
                        "    nodeData.append(nodeInputNames)\n"
                        "    nodeOutputs=[x for x in i.outputs()]\n"
                        "    nodeOutputNames=[x.debugName() for x in nodeOutputs]\n"
                        "    nodeData.append(nodeOutputNames)\n"
                        "    modelData.append(nodeData)",fGlobalNS,fLocalNS);
            PyRunString("print(modelData)",fGlobalNS,fLocalNS);
            Py_ssize_t modelIterator, modelSize;
            PyObject* pModel =PyDict_GetItemString(fLocalNS,"modelData");
            PyObject* node;
            modelSize = PyList_Size(pModel);

            for(modelIterator=0;modelIterator<modelSize;++modelIterator){
                node=PyList_GetItem(pModel,modelIterator);
                std::string type(PyStringAsString(PyList_GetItem(node,0)));

                if(NType.find(toLower(type))==NType.end())
                    throw std::runtime_error("Layer error: TMVA SOFIE does not yet suppport layer type"+type);

                PyObject* attributes=PyList_GetItem(node,1);
                PyObject* inputs=PyList_GetItem(node,2);
                PyObject* outputs=PyList_GetItem(node,3);


                switch(NType.find(toLower(type))->second){
                    case NodeType::GEMM : {
                        float attr_alpha = (float)(PyFloat_AsDouble(PyDict_GetItemString(attributes,"alpha")));
                        float attr_beta = (float)(PyFloat_AsDouble(PyDict_GetItemString(attributes,"beta")));
                        int_t attr_transA;
                        int_t attr_transB;

                        if(PyDict_Contains(attributes,PyUnicode_FromString("transB"))){
                             attr_transB=(int_t)(PyLong_AsLong(PyDict_GetItemString(attributes,"transB")));
                             attr_transA=!attr_transB;
                        }
                        else{
                            attr_transA=(int_t)(PyLong_AsLong(PyDict_GetItemString(attributes,"transA")));
                            attr_transB=!attr_transA;
                        }
                        std::unique_ptr<ROperator> op;
                        op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, PyStringAsString(PyList_GetItem(inputs,0)), PyStringAsString(PyList_GetItem(inputs,1)), PyStringAsString(PyList_GetItem(inputs,2)), PyStringAsString(PyList_GetItem(outputs,0))));
                        rmodel.AddOperator(std::move(op));
                        break;
                        }

                    case NodeType::RELU : {
                        std::unique_ptr<ROperator> op;
                        op.reset(new ROperator_Relu<float>(PyStringAsString(PyList_GetItem(inputs,0)), PyStringAsString(PyList_GetItem(outputs,0))));
                        rmodel.AddOperator(std::move(op));
                        break;
                    }

                    case NodeType::TRANSPOSE:{
                        std::unique_ptr<ROperator> op;
                        std::vector<int_t> attr_perm;
                        PyObject* permute=PyDict_GetItemString(attributes,"perm");
                        for(Py_ssize_t permIter=0; permIter<PyList_Size(permute);++permIter){
                            attr_perm.push_back((int_t)PyLong_AsLong(PyList_GetItem(permute,permIter)));
                            }
                        op.reset(new ROperator_Transpose<float>(attr_perm, PyStringAsString(PyList_GetItem(inputs,0)), PyStringAsString(PyList_GetItem(outputs,0))));
                        rmodel.AddOperator(std::move(op));
                        break;
                        }

                    default:
                        throw std::runtime_error("Node Error: TMVA SOFIE does not yet support node type " + type);

                }
            }


            Py_XDECREF(node);
            Py_XDECREF(pModel);


            //Extracting model weights to add the initialized tensors to the RModel
            PyRunString("weightNames=[k for k in graph[1].keys()]",fGlobalNS,fLocalNS);
            PyRunString("weights=[v.numpy() for v in graph[1].values()]",fGlobalNS,fLocalNS);
            PyObject* weightNames= PyDict_GetItemString(fLocalNS,"weightNames");
            PyObject* weightTensors= PyDict_GetItemString(fLocalNS,"weights");
            PyObject* weightTensor;
            for(Py_ssize_t weightIter=0; weightIter<PyList_Size(weightNames);++weightIter){
                weightTensor= PyList_GetItem(weightTensors,weightIter);
                std::string weightName(PyStringAsString(PyList_GetItem(weightNames,weightIter)));

                //Converting the numpy array object to RTensor
                RTensor<float> value=getArray(weightTensor);
                std::shared_ptr<void> data(malloc(value.GetSize() * sizeof(float)), free);
                std::memcpy(data.get(),value.GetData(),value.GetSize() * sizeof(float));
                rmodel.AddInitializedTensor(weightName, ETensorType::FLOAT,value.GetShape(), data);
            }

            Py_XDECREF(weightNames);
            Py_XDECREF(weightTensors);
            Py_XDECREF(weightTensor);


            //Extracting Input tensor info
            PyRunString("inputs=[x for x in model.graph.inputs()]",fGlobalNS,fLocalNS);
            PyRunString("inputs=inputs[1:]",fGlobalNS,fLocalNS);
            PyRunString("inputNames=[x.debugName() for x in inputs]",fGlobalNS,fLocalNS);
            PyObject* pInputs= PyDict_GetItemString(fLocalNS,"inputNames");
            for(Py_ssize_t inputIter=0; inputIter<PyList_Size(pInputs);++inputIter){
                std::string inputName(PyStringAsString(PyList_GetItem(pInputs,inputIter)));
                std::vector<size_t>inputShape=inputShapes[inputIter];
                rmodel.AddInputTensorInfo(inputName, ETensorType::FLOAT, inputShape);
            }

            Py_XDECREF(pInputs);


            //Extracting output tensor names
            PyRunString("outputs=[x for x in graph[0].outputs()]",fGlobalNS,fLocalNS);
            PyRunString("outputNames=[x.debugName() for x in outputs]",fGlobalNS,fLocalNS);
            PyObject* pOutputs= PyDict_GetItemString(fLocalNS,"outputNames");
            std::vector<std::string> outputNames;
            for(Py_ssize_t outputIter = 0; outputIter < PyList_Size(pOutputs);++outputIter){
                outputNames.push_back(UTILITY::Clean_name(PyStringAsString(PyList_GetItem(pOutputs,outputIter))));
                }
            rmodel.AddOutputTensorNameList(outputNames);

            Py_XDECREF(pOutputs);
            Py_XDECREF(fLocalNS);
            Py_XDECREF(fGlobalNS);
            Py_XDECREF(main);

            return rmodel;



        }

        default:
        throw std::runtime_error("Type Error: TMVA SOFIE does not yet support the provided data type");


    }
}
}
}
}
}
