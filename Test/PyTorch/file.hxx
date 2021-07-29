//Code generated automatically by TMVA for Inference of Model file [file.onnx] at [Thu Jul 29 15:31:51 2021] 
#include<algorithm>
#include<vector>
namespace TMVA_SOFIE_file{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_0bias[12] = {0.254049063, -0.00231359177, -0.675640047, -0.246749759, 0.000459161121, -0.327370226, -0.72559768, -0.117427371, 0.220769405, 0.319470078, 0.0186114535, 0.387177944};
float tensor_0weight[12] = {-1.11625803, -0.800215662, 0.720228851, -0.351409644, 0.157715231, -0.638390899, 0.773497581, 0.125026822, -0.235511839, 0.884459972, 0.366095781, 0.440946996};
float tensor_4[144];
float tensor_3[144];
std::vector<float> infer(float* tensor_input1){
	char op_0_transA = 'n';
	char op_0_transB = 't';
	int op_0_m = 12;
	int op_0_n = 12;
	int op_0_k = 1;
	float op_0_alpha = 1;
	float op_0_beta = 1;
	int op_0_lda = 1;
	int op_0_ldb = 1;
	std::copy(tensor_0bias, tensor_0bias + 12, tensor_3);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_0weight, &op_0_ldb, tensor_input1, &op_0_lda, &op_0_beta, tensor_3, &op_0_n);
	for (int id = 0; id < 144 ; id++){
		tensor_4[id] = ((tensor_3[id] > 0 )? tensor_3[id] : 0);
	}
	std::vector<float> ret (tensor_4, tensor_4 + sizeof(tensor_4) / sizeof(tensor_4[0]));
	return ret;
}
} //TMVA_SOFIE_file
