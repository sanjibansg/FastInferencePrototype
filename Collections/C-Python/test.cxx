
#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"


using namespace TMVA::Experimental::SOFIE;
void ConvertToRoot(){
  cout<<"Setting up Convertor for PyKeras"<<endl;
  int UseTFKeras=0;
  if(!UseTFKeras){
     cout<<"Converter only supports using TensorFlow backend for now"<<endl;
     cout<<kInfo<<"Method have to implement TensorFlow for conversion"<<endl;
     return;
  }

  cout<<"Converter static ran!";

}

int main()
{
ConvertToRoot();
return 0;
}
