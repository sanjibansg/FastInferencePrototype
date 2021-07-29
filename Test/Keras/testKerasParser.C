#include "TMVA/RModelParser_Keras.h"
#include "KerasFunctionalModel.hxx"


using namespace TMVA::Experimental;

int testKerasParser() {

float input[]={0.86675036, 0.58135283, 0.09423006, 0.9223447 , 0.8036846 , 0.64241505, 0.6127492 , 0.26899576, 0.638404  , 0.75192666,0.8923881 , 0.03006351, 0.20690072, 0.1505388 , 0.01444924,0.5005646 };

std::cout<<"Testing PyTorch Keras for nn.Module model\n";
std::vector<float> outModule = TMVA_SOFIE_KerasModelFunctional::infer(input);
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

SOFIE::PyRunString("import tensorflow as tf",fGlobalNS,fLocalNS);
SOFIE::PyRunString("model=tf.keras.models.load_model('KerasModelFunctional.h5')",fGlobalNS,fLocalNS);
SOFIE::PyRunString("ip=tf.reshape(tf.convert_to_tensor([0.86675036, 0.58135283, 0.09423006, 0.9223447 , 0.8036846 , 0.64241505, 0.6127492 , 0.26899576, 0.638404  , 0.75192666,0.8923881 , 0.03006351, 0.20690072,0.1505388 , 0.01444924,0.5005646]),[1,16])",fGlobalNS,fLocalNS);
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
