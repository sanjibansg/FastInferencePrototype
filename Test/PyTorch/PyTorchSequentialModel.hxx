//Code generated automatically by TMVA for Inference of Model file [PyTorchModelSequential.pt] at [Fri Jul 30 14:48:50 2021] 
#include<algorithm>
#include<vector>
namespace TMVA_SOFIE_PyTorchModelSequential{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
}//BLAS
float tensor_0weight[24] = {-0.102689922, -0.0337643921, 0.42570278, -0.115919657, 0.259498179, 0.450492561, 0.391875088, 0.34471029, -0.160325885, 0.468532026, -0.224604845, 0.195395291, 0.466508865, -0.277709663, -0.122888774, -0.161971793, -0.0661229193, -0.366587251, -0.349220008, -0.0955176428, -0.17071569, -0.0883811712, -0.301739037, -0.178474665};
float tensor_0bias[12] = {-0.0537942611, 0.130939782, -0.161948621, -0.0492868125, -0.261009991, -0.434376359, -0.0537942611, 0.130939782, -0.161948621, -0.0492868125, -0.261009991, -0.434376359};
float tensor_4[12];
float tensor_3[12];
std::vector<float> infer(float* tensor_input1){
	float op_0_alpha = 1;
	float op_0_beta = 1;
	char op_0_transA = 'n';
	char op_0_transB = 't';
	int op_0_m = 2;
	int op_0_n = 6;
	int op_0_k = 4;
	int op_0_lda = 4;
	int op_0_ldb = 4;
	std::copy(tensor_0bias, tensor_0bias + 12, tensor_3);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_0weight, &op_0_ldb, tensor_input1, &op_0_lda, &op_0_beta, tensor_3, &op_0_n);
	for (int id = 0; id < 12 ; id++){
		tensor_4[id] = ((tensor_3[id] > 0 )? tensor_3[id] : 0);
	}
	std::vector<float> ret (tensor_4, tensor_4 + sizeof(tensor_4) / sizeof(tensor_4[0]));
	return ret;
}
} //TMVA_SOFIE_PyTorchModelSequential
