//Code generated automatically by TMVA for Inference of Model file [modelfin.h5] at [Sun Jul  4 18:36:26 2021] 
#include<algorithm>
#include<vector>
namespace TMVA_SOFIE_modelfin{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_dense2bias0[40] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
float tensor_dense2kernel0[40] = {-0.326641142, -0.444902748, -0.211648524, 0.274051428, 0.135033488, 0.0561433434, -0.225645423, 0.144230187, -0.177695245, 0.149285197, -0.515414655, 0.163230658, -0.394368857, -0.389713049, 0.519566596, -0.233711064, -0.0737835169, -0.114104122, 0.276172101, 0.252932668, 0.0826721787, 0.188759446, 0.243679345, 0.183377385, 0.347965717, 0.478358567, -0.436329991, 0.00757455826, -0.008669734, 0.2997244, 0.413150728, -0.329927921, -0.202308148, -0.480605632, -0.184320688, 0.278069496, -0.326304555, -0.393903941, 0.227971554, -0.0318275094};
float tensor_dense2BiasAdd0[40];
std::vector<float> infer(float* tensor_input4){
	char op_0_transA = 't';
	char op_0_transB = 'n';
	int op_0_m = 2;
	int op_0_n = 20;
	int op_0_k = 1;
	float op_0_alpha = 1;
	float op_0_beta = 1;
	int op_0_lda = 2;
	int op_0_ldb = 20;
	std::copy(tensor_dense2bias0, tensor_dense2bias0 + 40, tensor_dense2BiasAdd0);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_dense2kernel0, &op_0_ldb, tensor_input4, &op_0_lda, &op_0_beta, tensor_dense2BiasAdd0, &op_0_n);
	std::vector<float> ret (tensor_dense2BiasAdd0, tensor_dense2BiasAdd0 + sizeof(tensor_dense2BiasAdd0) / sizeof(tensor_dense2BiasAdd0[0]));
	return ret;
}
} //TMVA_SOFIE_modelfin
