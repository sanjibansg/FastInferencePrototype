#include "TMVA/RModelParser_Keras.h"
#include "KerasSequentialModel.hxx"


using namespace TMVA::Experimental;

int testKerasParser() {

float input[]={0.4067344 , 0.70415358, 0.15035029, 0.25478038, 0.61434414, 0.34671188, 0.77180414, 0.78665889, 0.17410079, 0.89511033, 0.80029258, 0.77365451, 0.43891627, 0.20024058, 0.81597867, 0.23941791};

std::cout<<"Testing PyTorch Keras for nn.Module model\n";
std::vector<float> outModule = TMVA_SOFIE_KerasModelSequential::infer(input);
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

SOFIE::PyRunString("from keras.models import load_model",fGlobalNS,fLocalNS);
SOFIE::PyRunString("import numpy",fGlobalNS,fLocalNS);
SOFIE::PyRunString("model=load_model('KerasModelSequential.h5')",fGlobalNS,fLocalNS);
SOFIE::PyRunString("ip=numpy.array([0.4067344 , 0.70415358, 0.15035029, 0.25478038, 0.61434414, 0.34671188, 0.77180414, 0.78665889, 0.17410079, 0.89511033, 0.80029258, 0.77365451, 0.43891627, 0.20024058, 0.81597867, 0.23941791]).reshape(4,4)",fGlobalNS,fLocalNS);
SOFIE::PyRunString("op=model(ip).numpy()",fGlobalNS,fLocalNS);
PyObject* output = PyDict_GetItemString(fLocalNS,"op");
RTensor<float> value=SOFIE::getArray(output);

float* valOp=(float*)value.GetData();
std::vector<float>values{valOp,valOp+value.GetSize()};


for (auto i = outModule.begin(); i != outModule.end(); ++i)
    std::cout << *i << ' ';
std::cout<<"\n\noutMod done \n\n";



for (auto i = values.begin(); i != values.end(); ++i)
    std::cout << *i << ' ';
return 0;
}


int main(){
    int err= testKerasParser();
    return err;
}
