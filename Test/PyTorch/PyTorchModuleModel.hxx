//Code generated automatically by TMVA for Inference of Model file [PyTorchModelModule.pt] at [Thu Jul 29 13:55:45 2021] 
#include<algorithm>
#include<vector>
namespace TMVA_SOFIE_PyTorchModelModule{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_fc1weight[12] = {-0.901524067, -0.0270733833, -0.399220198, -0.0228905678, 0.406450748, 0.515890837, 0.114710212, -0.543226838, -0.705387235, -0.527284265, 0.788671017, -0.0495590605};
float tensor_fc1bias[12] = {-0.882822871, -0.760092258, 0.192892939, -0.859875321, -0.338998526, -0.466079116, -0.292136431, -0.708631039, -0.659826636, -0.795306087, -0.0266577546, 0.0412917249};
float tensor_4[144];
float tensor_5[144];
float tensor_3[144];
std::vector<float> infer(float* tensor_x1){
	char op_0_transA = 'n';
	char op_0_transB = 't';
	int op_0_m = 12;
	int op_0_n = 12;
	int op_0_k = 1;
	float op_0_alpha = 1;
	float op_0_beta = 1;
	int op_0_lda = 1;
	int op_0_ldb = 1;
	std::copy(tensor_fc1bias, tensor_fc1bias + 12, tensor_3);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_fc1weight, &op_0_ldb, tensor_x1, &op_0_lda, &op_0_beta, tensor_3, &op_0_n);
	for (int id = 0; id < 144 ; id++){
		tensor_4[id] = ((tensor_3[id] > 0 )? tensor_3[id] : 0);
	}
	for (int id = 0; id < 144 ; id++){
		 tensor_5[id / 12 % 12 * 1 + id / 1 % 12 * 12] = tensor_4[id];
	}
	std::vector<float> ret (tensor_5, tensor_5 + sizeof(tensor_5) / sizeof(tensor_5[0]));
	return ret;
}
} //TMVA_SOFIE_PyTorchModelModule
