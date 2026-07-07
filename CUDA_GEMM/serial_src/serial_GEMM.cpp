#include <iostream>
#include <vector>

/*
	这是串行的矩阵乘法算法
	用于后续并行计算优化
*/

bool serialGEMM(std::vector<std::vector<float>>& a, std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c) {
	int row_a = a.size(); int col_a = a[0].size();
	int row_b = b.size(); int col_b = b[0].size();
	if (col_a != row_b) {
		return false;
	}
	// i是a的行数遍历
	for (int i = 0; i < row_a; ++i) {
		// j是b的列数遍历
		for (int j = 0; j < col_b; ++j) {
			// k是a的列数==b的行数
			// a的列数和b的行数应当相同
			float prod = 0;
			for (int k = 0; k < col_a; ++k) {
				prod += a[i][k] * b[k][j];
			}
			c[i][j] = prod;
		}
	}
	return true;
}

int main() {
	// 定义行列数
	/*const int M = rand();
	const int K = rand();
	const int N = rand();*/
	const int M = 3;
	const int K = 4;
	const int N = 5;
	// M*K · K*N = M*N
	const int row_a = M, col_a = K;
	const int row_b = K, col_b = N;
	const int row_c = M, col_c = N;
	// a矩阵，b矩阵作为运算，c矩阵存储结果
	std::vector<std::vector<float>> a(row_a, std::vector<float>(col_a));
	std::vector<std::vector<float>> b(row_b, std::vector<float>(col_b));
	std::vector<std::vector<float>> c(row_c, std::vector<float>(col_c));

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

	// 计算两个矩阵运算结果
	bool calculateStatus = serialGEMM(a, b, c);

	if (calculateStatus) {
		std::cout << "计算成功！" << std::endl;
		// 输出结果
			std::cout << "A矩阵是：" << std::endl;
		for (auto& r : a) {
			int cnt = 0;
			for (auto& n : r) {
				std::cout << n << "	";
				cnt++;
				if (cnt == col_a) {
					std::cout << std::endl;
				}
			}
		}
		std::cout << "B矩阵是：" << std::endl;
		for (auto& r : b) {
			int cnt = 0;
			for (auto& n : r) {
				std::cout << n << "	";
				cnt++;
				if (cnt == col_b) {
					std::cout << std::endl;
				}
			}
		}

		std::cout << "两矩阵乘积计算得到：" << std::endl;
		for (auto& r : c) {
			int cnt = 0;
			for (auto& n : r) {
				std::cout << n << "	";
				cnt++;
				if (cnt == col_c) {
					std::cout << std::endl;
				}
			}
		}
	}
	else {
		std::cout << "计算失败！请检查。" << std::endl;
	}
}