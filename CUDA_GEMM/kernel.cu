//--UTF-8--
#include <iostream>
#include <vector>
#include <chrono>  // 高精度计时
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================
// 选择GPU设备
// ============================================
bool selectGPU(int deviceId = 0) {
	int deviceCount;
	cudaError_t status;

	// 1. 获取CUDA设备数量
	status = cudaGetDeviceCount(&deviceCount);
	if (status != cudaSuccess) {
		std::cerr << "获取设备数量失败：" << cudaGetErrorString(status) << std::endl;
		return false;
	}

	if (deviceCount == 0) {
		std::cerr << "没有找到CUDA设备！" << std::endl;
		return false;
	}

	std::cout << "找到 " << deviceCount << " 块CUDA显卡" << std::endl;

	// 2. 显示所有设备信息
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "  设备 " << i << ": " << prop.name << std::endl;
		std::cout << "    计算能力: " << prop.major << "." << prop.minor << std::endl;
		std::cout << "    显存: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
	}

	// 3. 选择设备
	if (deviceId >= deviceCount) {
		std::cerr << "设备 " << deviceId << " 不存在，使用设备0" << std::endl;
		deviceId = 0;
	}

	status = cudaSetDevice(deviceId);
	if (status != cudaSuccess) {
		std::cerr << "设置设备 " << deviceId << " 失败：" << cudaGetErrorString(status) << std::endl;
		return false;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, deviceId);
	std::cout << "✅ 使用设备: " << prop.name << std::endl;

	return true;
}

// ============ 此部分为并行算法 ============
__global__ void gemmKernel(float* A, float* B, float* C, int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N) {
		float sum = 0.0f;
		for (int k = 0; k < K; k++) {
			sum += A[row * K + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
}

bool cudaGEMM(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c,
	int M, int N, int K, float& kernelTime) {
	size_t bytesA = M * K * sizeof(float);
	size_t bytesB = K * N * sizeof(float);
	size_t bytesC = M * N * sizeof(float);

	float* dev_A = nullptr;
	float* dev_B = nullptr;
	float* dev_C = nullptr;
	cudaError_t status;

	// 分配GPU内存
	status = cudaMalloc((void**)&dev_A, bytesA);
	if (status != cudaSuccess) {
		std::cerr << "A矩阵显存分配失败！" << cudaGetErrorString(status) << std::endl;
		goto ERROR;
	}

	status = cudaMalloc((void**)&dev_B, bytesB);
	if (status != cudaSuccess) {
		std::cerr << "B矩阵分配显存失败！" << cudaGetErrorString(status) << std::endl;
		goto ERROR;
	}

	status = cudaMalloc((void**)&dev_C, bytesC);
	if (status != cudaSuccess) {
		std::cerr << "C矩阵分配显存失败！" << cudaGetErrorString(status) << std::endl;
		goto ERROR;
	}

	// 拷贝CPU数据到GPU
	status = cudaMemcpy(dev_A, a.data(), bytesA, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		std::cerr << "A矩阵数据拷贝失败！" << cudaGetErrorString(status) << std::endl;
		goto ERROR;
	}
	status = cudaMemcpy(dev_B, b.data(), bytesB, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		std::cerr << "B矩阵数据拷贝失败！" << cudaGetErrorString(status) << std::endl;
		goto ERROR;
	}

	// 配置并启动核函数
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(
		// 向上取整写法
		(N + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(M + threadsPerBlock.y - 1) / threadsPerBlock.y
	);

	// 创建cuda事件，精确计时
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// 记录开始
	cudaEventRecord(start, 0);

	// 启动核函数
	gemmKernel << <blocksPerGrid, threadsPerBlock >> > (dev_A, dev_B, dev_C, M, N, K);

	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cerr << "核函数启动失败！" << cudaGetErrorString(status) << std::endl;
		goto ERROR;
	}

	// 等待GPU完成运算
	cudaDeviceSynchronize();

	// 记录结束时间
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	// 计算核函数执行时间
	cudaEventElapsedTime(&kernelTime, start, end);

	// 销毁事件
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	// 拷贝结果回CPU
	status = cudaMemcpy(c.data(), dev_C, bytesC, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		std::cerr << "结果矩阵C拷贝回CPU失败！" << cudaGetErrorString(status) << std::endl;
		goto ERROR;
	}

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return true;

	// 失败集中清理
ERROR:
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return false;
}

// ============ 此部分是串行算法 ============
bool serialGEMM(std::vector<std::vector<float>>& a, std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c,
	int M, int N, int K, double& elapsedTime) {
	// 开始计时
	auto start = std::chrono::high_resolution_clock::now();

	// i是a的行数遍历
	for (int i = 0; i < M; ++i) {
		// j是b的列数遍历
		for (int j = 0; j < N; ++j) {
			// k是a的列数==b的行数
			// a的列数和b的行数应当相同
			float prod = 0.0;
			for (int k = 0; k < K; ++k) {
				prod += a[i][k] * b[k][j];
			}
			c[i][j] = prod;
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	elapsedTime = std::chrono::duration<double, std::milli>(end - start).count();

	return true;
}

// 二维vector压缩为一维vector
std::vector<float> flatten2D(std::vector<std::vector<float>>& a) {
	std::vector<float> ret;
	for (auto& r : a) {
		for (auto& n : r) {
			ret.push_back(n);
		}
	}
	return ret;
}

int main() {
	// ⭐ 首先选择GPU
	if (!selectGPU(0)) {
		return 1;
	}
	// 定义行列数
	/*
	小数组验证正确性
	const int M = 3;
	const int K = 4;
	const int N = 5;
	*/
	const int M = 32768;
	const int K = 32768;
	const int N = 32768;
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

	// ========= 此部分为串行计算 =========
	// 计时
	double serialTime = 0.0;
	bool serialStatus = false;
	bool bigMatrix = true;// 大数组模式
	// 计算两个矩阵运算结果
	if (!bigMatrix) {
		serialStatus = serialGEMM(a, b, c, M, N, K, serialTime);
	}
	else {
		serialStatus = true;
	}
	

	if (serialStatus) {
		if (bigMatrix) {
			std::cout << "大数组模式，仅作并行运算！" << std::endl;
		}
		else {
			std::cout << "串行计算成功！" << std::endl;
		}
		// 输出结果
		/*std::cout << "A矩阵是：" << std::endl;
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
		}*/

		//std::cout << "串行计算时间为：" << serialTime << "ms" << std::endl;
	}
	else {
		std::cerr << "串行计算失败！请检查。" << std::endl;
		return -1;
	}

	// ========= 此部分为并行计算 =========
	bool parallelStatus;
	std::vector<float> A;
	std::vector<float> B;
	std::vector<float> C;
	float parallelTime = 0.0;
	if (serialStatus || bigMatrix) {// 串行计算成功才开始算，不然无法验证正确性（或者大数组模式直接开始计算）
		// 一维化
		A = flatten2D(a);
		B = flatten2D(b);
		C = flatten2D(c);// 后续C会被修改，此处仅作内存分配
		// CUDA计算
		parallelStatus = cudaGEMM(A, B, C, M, N, K, parallelTime);
		if (parallelStatus) {
			std::cout << "并行计算成功！" << std::endl;
		}
		else {
			std::cerr << "并行计算失败！请检查。" << std::endl;
			return -2;
		}
	}

	// 检查使用的设备
	int currentDevice;
	cudaGetDevice(&currentDevice);
	std::cout << "当前设备: " << currentDevice << std::endl;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, currentDevice);
	std::cout << "设备名称: " << prop.name << std::endl;

	// 对比结果
	if (serialStatus && parallelStatus) {
		if (!bigMatrix) {
			if (flatten2D(c) == C) {
				std::cout << "串行计算与并行计算结果一致，计算成功！" << std::endl;
				std::cout << "串行计算时间为：" << serialTime << "ms" << std::endl;
				std::cout << "并行计算时间为：" << parallelTime << "ms" << std::endl;
			}
			else {
				std::cerr << "计算结果出错！请检查。" << std::endl;
			}
		}
		else {
			std::cout << "大数组运算时间为：" << parallelTime << "ms" << std::endl;
		}
		
	}

	return 0;
}
