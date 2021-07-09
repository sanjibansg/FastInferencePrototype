//Code generated automatically by TMVA for Inference of Model file [modelfull.h5] at [Sun Jul  4 19:03:07 2021] 
#include<algorithm>
#include<vector>
namespace TMVA_SOFIE_modelfull{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_dense1bias0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
float tensor_dense1kernel0[8] = {0.706400156, -0.711347878, -0.0925500393, 0.34772408, -0.216604531, -0.000303864479, -0.335873276, -0.601884365};
float tensor_densebias0[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
float tensor_densekernel0[32] = {-0.62619698, -0.471547186, -0.212853372, 0.0233342052, -0.0283955336, 0.137989283, -0.661990285, 0.171279728, 0.164644778, -0.673859298, -0.601808906, 0.112333477, 0.565500081, 0.605041802, 0.683745086, 0.467251718, 0.350034773, 0.405445635, -0.500826597, 0.676145136, 0.44021064, -0.522339702, -0.687115669, 0.371661484, -0.479177952, 0.0483281016, 0.550504386, 0.357718527, 0.619001091, -0.260804683, 0.472887814, -0.310393333};
float tensor_dense1Relu0[8];
float tensor_dense1gemm[8];
float tensor_denseBiasAdd0[32];
std::vector<float> infer(float* tensor_input1){
	char op_0_transA = 't';
	char op_0_transB = 'n';
	int op_0_m = 4;
	int op_0_n = 8;
	int op_0_k = 1;
	float op_0_alpha = 1;
	float op_0_beta = 1;
	int op_0_lda = 4;
	int op_0_ldb = 8;
	std::copy(tensor_densebias0, tensor_densebias0 + 32, tensor_denseBiasAdd0);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_densekernel0, &op_0_ldb, tensor_input1, &op_0_lda, &op_0_beta, tensor_denseBiasAdd0, &op_0_n);
	char op_1_transA = 't';
	char op_1_transB = 'n';
	int op_1_m = 8;
	int op_1_n = 1;
	int op_1_k = 4;
	float op_1_alpha = 1;
	float op_1_beta = 1;
	int op_1_lda = 8;
	int op_1_ldb = 1;
	std::copy(tensor_dense1bias0, tensor_dense1bias0 + 8, tensor_dense1gemm);
	BLAS::sgemm_(&op_1_transB, &op_1_transA, &op_1_n, &op_1_m, &op_1_k, &op_1_alpha, tensor_dense1kernel0, &op_1_ldb, tensor_denseBiasAdd0, &op_1_lda, &op_1_beta, tensor_dense1gemm, &op_1_n);
	for (int id = 0; id < 8 ; id++){
		tensor_dense1Relu0[id] = ((tensor_dense1gemm[id] > 0 )? tensor_dense1gemm[id] : 0);
	}
	std::vector<float> ret (tensor_dense1Relu0, tensor_dense1Relu0 + sizeof(tensor_dense1Relu0) / sizeof(tensor_dense1Relu0[0]));
	return ret;
}
} //TMVA_SOFIE_modelfull
