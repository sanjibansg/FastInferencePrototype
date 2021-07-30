//Code generated automatically by TMVA for Inference of Model file [PyTorchModelModule.pt] at [Fri Jul 30 14:48:33 2021] 
#include<algorithm>
#include<vector>
namespace TMVA_SOFIE_PyTorchModelModule{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
}//BLAS
float tensor_fc1weight[72] = {0.406532109, 0.116485804, 0.102107048, 0.144761562, -0.387347162, -0.163734153, 0.163307279, -0.0642060637, -0.139611617, -0.354690492, 0.239859521, -0.251577854, -0.377916664, -0.0776686892, 0.238153756, -0.258211672, -0.0763916448, -0.0836719126, 0.508134007, 0.493486553, 0.0411019959, -0.505379856, 0.231491521, -0.0289624166, -0.29788962, -0.0247434173, 0.11643149, 0.176735446, 0.34592399, 0.204855427, -0.206473604, -0.0388607867, -0.151029631, 0.174633861, 0.333050787, 0.0235689376, 0.0177842323, -0.142235219, -0.0750631392, 0.214810252, -0.189897045, 0.211665735, 0.0813454017, -0.0571494997, 0.00992859807, 0.156483844, -0.30277589, -0.169037014, -0.250545114, 0.366617054, -0.147707134, 0.367898703, 0.161184788, 0.369491279, -0.215061739, -0.146556884, -0.19443953, -0.254338562, 0.253289223, 0.0235979222, -0.0589905195, 0.26075533, -0.274694115, -0.406268656, -0.385340422, 0.31349501, -0.20183982, 0.400007635, 0.149284005, -0.0545759909, -0.335794687, -0.161013618};
float tensor_fc1bias[24] = {-0.277438968, -0.206046134, 0.0248444099, -0.0714089796, -0.0640285313, -0.271402776, -0.177777305, -0.121993884, -0.31619066, -0.436826706, -0.405816644, 0.157974467, -0.277438968, -0.206046134, 0.0248444099, -0.0714089796, -0.0640285313, -0.271402776, -0.177777305, -0.121993884, -0.31619066, -0.436826706, -0.405816644, 0.157974467};
float tensor_4[24];
float tensor_5[24];
float tensor_3[24];
std::vector<float> infer(float* tensor_x1){
	float op_0_alpha = 1;
	float op_0_beta = 1;
	char op_0_transA = 'n';
	char op_0_transB = 't';
	int op_0_m = 2;
	int op_0_n = 12;
	int op_0_k = 6;
	int op_0_lda = 6;
	int op_0_ldb = 6;
	std::copy(tensor_fc1bias, tensor_fc1bias + 24, tensor_3);
	BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_fc1weight, &op_0_ldb, tensor_x1, &op_0_lda, &op_0_beta, tensor_3, &op_0_n);
	for (int id = 0; id < 24 ; id++){
		tensor_4[id] = ((tensor_3[id] > 0 )? tensor_3[id] : 0);
	}
	for (int id = 0; id < 24 ; id++){
		 tensor_5[id / 12 % 2 * 1 + id / 1 % 12 * 2] = tensor_4[id];
	}
	std::vector<float> ret (tensor_5, tensor_5 + sizeof(tensor_5) / sizeof(tensor_5[0]));
	return ret;
}
} //TMVA_SOFIE_PyTorchModelModule
