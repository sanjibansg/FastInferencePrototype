#include "TMVA/RModelParser_PyTorch.h"
#include "PyTorchSequentialModel.hxx"

using namespace TMVA::Experimental;

int testPyTorchParser() {

float input[]={-1.2133,0.1405,0.3343,0.8215,-0.3429,1.6881,1.6557,0.5423,1.4364,-0.7098,0.0413,-0.9946};

std::cout<<"Testing PyTorch Parser for nn.Sequential model\n";
std::vector<float> outModule = TMVA_SOFIE_PyTorchModelSequential::infer(input);
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

SOFIE::PyRunString("import torch",fGlobalNS,fLocalNS);
SOFIE::PyRunString("model=torch.jit.load('PyTorchModelSequential.pt')",fGlobalNS,fLocalNS);
SOFIE::PyRunString("ip=torch.reshape(torch.FloatTensor([-1.2133,0.1405,0.3343,0.8215,-0.3429,1.6881,1.6557,0.5423,1.4364,-0.7098,0.0413,-0.9946]),(12,1))",fGlobalNS,fLocalNS);
SOFIE::PyRunString("op=model(ip).detach().numpy()",fGlobalNS,fLocalNS);
PyObject* output = PyDict_GetItemString(fLocalNS,"op");
RTensor<float> value=SOFIE::getArray(output);

float* valOp=(float*)value.GetData();
std::vector<float>values{valOp,valOp+value.GetSize()};


for (auto i = outModule.begin(); i != outModule.end(); ++i)
    std::cout << *i << ' ';
std::cout<<"\n\noutMod done \n\n";


cout<<value<<endl<<endl;
for (auto i = values.begin(); i != values.end(); ++i)
    std::cout << *i << ' ';
return 0;
}


int main(){
    int err= testPyTorchParser();
    return err;
}
