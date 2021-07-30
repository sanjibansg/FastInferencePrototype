//Code generated automatically by TMVA for Inference of Model file [KerasModelSequential.h5] at [Fri Jul 30 13:41:33 2021] 
#include<algorithm>
#include<vector>
namespace TMVA_SOFIE_KerasModelSequential{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
}//BLAS
float tensor_densebias0[24] = {0.00708308117, 0.00382814324, 0.0106032696, 0.0114305653, 0.00886055734, 0.00702357292, 0.00708308117, 0.00382814324, 0.0106032696, 0.0114305653, 0.00886055734, 0.00702357292, 0.00708308117, 0.00382814324, 0.0106032696, 0.0114305653, 0.00886055734, 0.00702357292, 0.00708308117, 0.00382814324, 0.0106032696, 0.0114305653, 0.00886055734, 0.00702357292};
float tensor_densekernel0[24] = {0.495408475, 0.185769424, -0.200769871, 0.173844844, 0.0629707575, 0.538041532, -0.15198487, 0.430591166, 0.418222576, 0.52102685, -0.328993261, -0.39821583, 0.739520252, 0.437441975, 0.406170189, 0.231026694, 0.468787014, 0.0331718065, -0.301836878, -0.295444608, -0.0123302825, -0.548944652, -0.0736432895, -0.38849014};
float tensor_activationRelu0[24];
float tensor_denseBiasAdd0[24];
std::vector<float> infer(float* tensor_denseinput){
	float op_0_alpha = 1;
	float op_0_beta = 1;
	char op_0_transA = 't';
	char op_0_transB = 'n';
	int op_0_m = 4;
	int op_0_n = 6;
	int op_0_k = 4;
	int op_0_lda = 4;
	int op_0_ldb = 6;
	std::copy(tensor_densebias0, tensor_densebias0 + 24, tensor_denseBiasAdd0);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_densekernel0, &op_0_ldb, tensor_denseinput, &op_0_lda, &op_0_beta, tensor_denseBiasAdd0, &op_0_n);
	for (int id = 0; id < 24 ; id++){
		tensor_activationRelu0[id] = ((tensor_denseBiasAdd0[id] > 0 )? tensor_denseBiasAdd0[id] : 0);
	}
	std::vector<float> ret (tensor_activationRelu0, tensor_activationRelu0 + sizeof(tensor_activationRelu0) / sizeof(tensor_activationRelu0[0]));
	return ret;
}
} //TMVA_SOFIE_KerasModelSequential
