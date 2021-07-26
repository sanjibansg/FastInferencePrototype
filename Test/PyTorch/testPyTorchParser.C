#include "TMVA/RModelParser_PyTorch.h"

using namespace TMVA::Experimental;


TString pythonExtract = "\
globals().update(locals())\n\
import torch\n\
from torch.onnx.utils import _model_to_graph\n\
from torch.onnx.symbolic_helper import _set_onnx_shape_inference\n\
\n\
model= torch.jit.load('%s')\n\
model.cpu()\n\
model.eval()\n\
\n\
ip=torch.rand(120,1)\n\
op=model(ip)\n\
\n\
_set_onnx_shape_inference(True)\n\
graph=_model_to_graph(model,ip,example_outputs=op)\n";


int testPyTorchParser(){

    std::string pythonSource="generatePyTorchModelSequential.py";
    std::string modelFileName="PyTorchModelSequential.pt";
    std::cout<<"Running Python script to generate model...\n";

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

    FILE* fp;
    fp = fopen(pythonSource.c_str(), "r");
    PyRun_SimpleFile(fp, pythonSource.c_str());
    Py_Finalize();
    std::vector<size_t> s1{120,1};
    std::vector<std::vector<size_t>>inputShape{s1};
    SOFIE::RModel model = SOFIE::PyTorch::Parse(modelFileName,inputShape);
    model.Generate();

    Py_Initialize();

    main = PyImport_AddModule("__main__");
    fGlobalNS = PyModule_GetDict(main);
    fLocalNS = PyDict_New();
    if (!fGlobalNS) {
        throw std::runtime_error("Can't init global namespace for Python");
        }
    if (!fLocalNS) {
        throw std::runtime_error("Can't init local namespace for Python");
        }

    //Extracting and testing intermediate tensor types & shapes
    std::cout<<"Testing Intermediate tensor shape and types...\n";
    SOFIE::PyRunString("print('hi')",fGlobalNS,fLocalNS);
    SOFIE::PyRunString(TString::Format(pythonExtract,modelFileName.c_str()),fGlobalNS,fLocalNS);
    std::cout<<"model extracted";
    SOFIE::PyRunString("modelData=[]",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("for i in graph[0].nodes():\n"
                   "    nodeData=[]\n"
                   "    nodeData.extend([x.debugName() for x in i.outputs()])\n"
                   "    nodeData.extend([x.type().scalarType() for x in i.outputs()])\n"
                   "    nodeData.extend([x.type().sizes() for x in i.outputs()])\n"
                   "    modelData.append(nodeData)",fGlobalNS,fLocalNS);
    std::cout<<"model";
    PyObject* pModel = PyDict_GetItemString(fLocalNS,"modelData");
    Py_ssize_t modelIterator, modelSize;
    modelSize = PyList_Size(pModel);
    PyObject *node,*shape;
    for(modelIterator=0;modelIterator<modelSize;++modelIterator){
        node=PyList_GetItem(pModel,modelIterator);
        std::string name(SOFIE::UTILITY::Clean_name(SOFIE::PyStringAsString(PyList_GetItem(node,0))));
        std::string type(SOFIE::PyStringAsString(PyList_GetItem(node,1)));
        shape=PyList_GetItem(node,1);

        if(model.CheckIfTensorAlreadyExist(name)){
            if(convertStringToType(SOFIE::dTypePyTorch,type) == model.GetTensorType(name)){
                std::vector<size_t>modelShape=model.GetTensorShape(name);
                std::vector<size_t>pModelShape;
                for(Py_ssize_t iter=0; iter<PyList_Size(shape);++iter){
                    pModelShape.push_back((size_t)(PyLong_AsLong(PyList_GetItem(shape,iter))));
                }
                if(modelShape!=pModelShape){
                    std::cout<<"[ERROR] Intermediate tensor "+name+" doesn't matches the tensor shape with the parsed tensor shape";
                    return 1;
                }
            }
            else{
                std::cout<<"[ERROR] Intermediate tensor "+name+"has type"+ConvertTypeToString(model.GetTensorType(name))+" which doesn't matches extracted type "+type;
                return 1;
            }
        }
        else{
            std::cout<<"[ERROR] Intermediate tensor "+name+" not found in RModel";
            return 1;
        }
    }

    //Extracting and testing initialized tensor data values
    std::cout<<"Testing Initialized tensor data\n";
    SOFIE::PyRunString("weightNames=[k for k in graph[1].keys()]",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("weights=[v.numpy() for v in graph[1].values()]",fGlobalNS,fLocalNS);
    PyObject* weightNames = PyDict_GetItemString(fLocalNS,"weightNames");
    PyObject* weightTensors = PyDict_GetItemString(fLocalNS,"weights");
    PyObject* weightTensor;
    int size=1;

    for(Py_ssize_t weightIter=0; weightIter<PyList_Size(weightNames);++weightIter){
        weightTensor= PyList_GetItem(weightTensors,weightIter);
        std::string weightName(SOFIE::UTILITY::Clean_name(SOFIE::PyStringAsString(PyList_GetItem(weightNames,weightIter))));
        std::vector<size_t>modelShape=model.GetTensorShape(weightName);
        for(auto& it: modelShape){
            size*=it;
        }
        float* initialzedTensor=(float*)(model.GetInitializedTensorData(weightName).get());
        std::vector<float>initlaizedTensorVector(initialzedTensor,initialzedTensor+size);
        RTensor<float>value = SOFIE::getArray(weightTensor);
        std::vector<float>weightTensorVector(value.GetData(),value.GetData()+size);
        if(initlaizedTensorVector!=weightTensorVector){
            std::cout<<"[ERROR] Initialized Tensors data and extracted model weights are not equal";
            return 1;
        }
    }
    return 0;
}

int main(){
    std::cout<<"Testing for PyTorch nn.Sequential...\n";
    int errSequential = testPyTorchParser();
    /*
    std::cout<<"Testing for PyTorch nn.Module...\n";
    int errModule = testPyTorchParser("generatePyTorchModelModule.py","PyTorchModelModule.pt");
    */
    return errSequential;
}
