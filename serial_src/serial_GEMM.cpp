#include <iostream>
#include <vector>

/*
	这是串行的矩阵乘法算法
	用于后续并行计算优化
*/

std::vector<std::vector<float>> serialGEMM(std::vector<std::vector<float>>& a, std::vector<std::vector<float>>& b) {
	int row_a = a.size(); int col_a = a[0].size();
	int row_b = b.size(); int col_b = b[0].size();
	std::vector<std::vector<float>> ret;
	for (int i = 0; i < row_a; ++i) {
		for (int j = 0; j < col_a; ++j) {

		}
	}
}

int main() {
	// 定义行列数
	const float M = rand();
	const float K = rand();
	const float N = rand();
	// M*K · K*N = M*N
	const int row_a = M, col_a = K;
	const int row_b = K, col_b = N;
	const int row_c = M, col_c = N;
	std::vector<std::vector<float>> a(row_a, std::vector(col_a));
	std::vector<std::vector<float>> b(row_b, std::vector(col_b));
	std::vector<std::vector<float>> c(row_c, std::vector(col_c));

	// 初始化a矩阵和b矩阵
	for (int i = 0; i < row_a; ++i) {
		for (int j = 0; j < col_a; ++j) {
			a[i][j] = rand();
		}
	}
	for (int i = 0; i < row_b; ++i) {
		for (int j = 0; j < col_b; ++j) {
			b[i][j] = rand();
		}
	}
}
