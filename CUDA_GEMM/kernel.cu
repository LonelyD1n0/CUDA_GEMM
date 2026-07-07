#include <iostream>
#include <vector>
#include <chrono>  // 高精度计时
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================
// 核函数：在GPU上执行矩阵乘法
// ============================================
__global__ void gemmKernel(float* A, float* B, float* C,
    int M, int N, int K)
{
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

// ============================================
// CUDA版本的GEMM封装函数
// ============================================
bool cudaGEMM(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c,
    int M, int N, int K, float& kernelTime)
{
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float* dev_A, * dev_B, * dev_C;
    cudaError_t status;

    // ========== 1. 分配GPU内存 ==========
    status = cudaMalloc((void**)&dev_A, bytesA);
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc dev_A failed: " << cudaGetErrorString(status) << std::endl;
        return false;
    }

    status = cudaMalloc((void**)&dev_B, bytesB);
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc dev_B failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(dev_A);
        return false;
    }

    status = cudaMalloc((void**)&dev_C, bytesC);
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc dev_C failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(dev_A);
        cudaFree(dev_B);
        return false;
    }

    // ========== 2. 拷贝数据到GPU ==========
    status = cudaMemcpy(dev_A, a.data(), bytesA, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy A failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
        return false;
    }

    status = cudaMemcpy(dev_B, b.data(), bytesB, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy B failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
        return false;
    }

    // ========== 3. 配置并启动核函数 ==========
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // ⭐ 创建CUDA事件用于精确计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start, 0);

    // 启动核函数
    gemmKernel << <blocksPerGrid, threadsPerBlock >> > (dev_A, dev_B, dev_C, M, N, K);

    // 检查核函数启动是否成功
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
        return false;
    }

    // 等待GPU完成计算
    cudaDeviceSynchronize();

    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算核函数执行时间（毫秒）
    cudaEventElapsedTime(&kernelTime, start, stop);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ========== 4. 拷贝结果回CPU ==========
    status = cudaMemcpy(c.data(), dev_C, bytesC, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy C failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
        return false;
    }

    // ========== 5. 清理GPU内存 ==========
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return true;
}

// ============================================
// 串行GEMM（返回执行时间）
// ============================================
bool serialGEMM(const std::vector<float>& a, const std::vector<float>& b,
    std::vector<float>& c, int M, int N, int K, double& elapsedTime)
{
    // ⭐ 使用chrono高精度计时
    auto start = std::chrono::high_resolution_clock::now();

    // 三重循环
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    elapsedTime = std::chrono::duration<double, std::milli>(end - start).count();

    return true;
}

// ============================================
// 打印矩阵
// ============================================
void printMatrix(const std::vector<float>& matrix, int rows, int cols,
    const std::string& name)
{
    std::cout << name << " (" << rows << "×" << cols << "):" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

// ============================================
// 计算GFLOPS（每秒十亿次浮点运算）
// ============================================
double computeGFLOPS(int M, int N, int K, double timeMs)
{
    // 总浮点运算次数：2 * M * N * K（乘法和加法各一次）
    double flops = 2.0 * M * N * K;
    double timeSec = timeMs / 1000.0;
    return flops / (timeSec * 1e9);
}

// ============================================
// 主函数
// ============================================
int main()
{
    // ========== 定义矩阵大小 ==========
    // 小矩阵用于验证正确性
    // const int M = 3, K = 4, N = 5;

    // 大矩阵用于性能测试
    const int M = 512;  // A的行数
    const int K = 512;  // A的列数 = B的行数
    const int N = 512;  // B的列数

    std::cout << "==================== GEMM性能测试 ====================" << std::endl;
    std::cout << "矩阵维度: " << M << "×" << K << " · "
        << K << "×" << N << " = " << M << "×" << N << std::endl;
    std::cout << "总元素数: " << M * N << " (" << (M * N) / 1024 << "K)" << std::endl;
    std::cout << "浮点运算量: " << 2.0 * M * N * K / 1e9 << " GFLOPS" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << std::endl;

    // ========== 分配内存 ==========
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_serial(M * N);
    std::vector<float> C_cuda(M * N);

    // 初始化A和B
    for (int i = 0; i < M * K; i++) {
        A[i] = rand() % 10;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = rand() % 10;
    }

    // ========== 串行版本 ==========
    std::cout << "【串行计算】" << std::endl;
    double serialTime;
    serialGEMM(A, B, C_serial, M, N, K, serialTime);
    double serialGFLOPS = computeGFLOPS(M, N, K, serialTime);
    std::cout << "  执行时间: " << serialTime << " ms" << std::endl;
    std::cout << "  性能: " << serialGFLOPS << " GFLOPS" << std::endl;
    std::cout << std::endl;

    // ========== CUDA版本 ==========
    std::cout << "【CUDA计算】" << std::endl;
    float kernelTime;
    bool success = cudaGEMM(A, B, C_cuda, M, N, K, kernelTime);

    if (!success) {
        std::cerr << "CUDA计算失败！" << std::endl;
        return 1;
    }
    double cudaGFLOPS = computeGFLOPS(M, N, K, kernelTime);
    std::cout << "  执行时间: " << kernelTime << " ms" << std::endl;
    std::cout << "  性能: " << cudaGFLOPS << " GFLOPS" << std::endl;
    std::cout << "  加速比: " << serialTime / kernelTime << "x" << std::endl;
    std::cout << std::endl;

    // ========== 验证正确性（仅小矩阵时打印） ==========
    if (M <= 16 && N <= 16) {
        std::cout << std::endl;
        printMatrix(A, M, K, "矩阵A");
        std::cout << std::endl;
        printMatrix(B, K, N, "矩阵B");
        std::cout << std::endl;
        printMatrix(C_serial, M, N, "串行结果C");
        std::cout << std::endl;
        printMatrix(C_cuda, M, N, "CUDA结果C");
    }

    // ========== 验证结果 ==========
    std::cout << std::endl;
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_cuda[i] - C_serial[i]) > 1e-5) {
            correct = false;
            std::cout << "❌ 验证失败！位置 " << i
                << ": CUDA=" << C_cuda[i]
                << ", 串行=" << C_serial[i] << std::endl;
                break;
        }
    }

    if (correct) {
        std::cout << "✅ 验证通过！CUDA结果与串行结果一致！" << std::endl;
    }
    else {
        std::cout << "❌ 验证失败！结果不一致！" << std::endl;
    }

    // ========== 性能总结 ==========
    std::cout << std::endl;
    std::cout << "==================== 性能总结 ====================" << std::endl;
    std::cout << "矩阵规模: " << M << "×" << K << " · " << K << "×" << N << std::endl;
    std::cout << "串行时间: " << serialTime << " ms" << std::endl;
    std::cout << "CUDA时间: " << kernelTime << " ms" << std::endl;
    std::cout << "加速比: " << serialTime / kernelTime << "x" << std::endl;
    std::cout << "==================================================" << std::endl;

    cudaDeviceReset();
    return 0;
}
