#include "TMVA/RModelParser_PyTorch.h"

using namespace TMVA::Experimental;

int generateROOTFromPyTorchModule(){
    Py_Initialize();
    std::cout<<"Generating PyTorch nn.Module model...\n";
    FILE* fp;
    fp = fopen("generatePyTorchModelModule.py", "r");
    PyRun_SimpleFile(fp, "generatePyTorchModelModule.py");
    TFile fileWrite("PyTorchModuleModel.root","RECREATE");
    std::vector<size_t> s1{12,1};
    std::vector<std::vector<size_t>> inputShape{s1};
    std::cout<<"Parsing saved PyTorch nn.Module model...\n";
    SOFIE::RModel model = SOFIE::PyTorch::Parse("PyTorchModelModule.pt",inputShape);
    model.Generate();
    std::cout<<"Writing PyTorch nn.Sequential RModel into a ROOT file...\n";
    model.Write("model");
    fileWrite.Close();
    TFile fileRead("PyTorchModuleModel.root","READ");
    SOFIE::RModel *modelPtr;
    fileRead.GetObject("model",modelPtr);
    fileRead.Close();
    modelPtr->OutputGenerated("PyTorchModuleModel.hxx");
    return 0;
}
