#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================
// 核函数：在GPU上执行矩阵乘法
// ============================================
__global__ void gemmKernel(float* A, float* B, float* C,
    int M, int N, int K)
{
    // 计算当前线程对应的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 越界保护：只处理有效的行和列
    if (row < M && col < N) {
        float sum = 0.0f;

        // 计算 C[row][col] = A[row][:] · B[:][col]
        for (int k = 0; k < K; k++) {
            // A[row * K + k]：A的第row行，第k列
            // B[k * N + col]：B的第k行，第col列
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// ============================================
// CUDA版本的GEMM封装函数
// ============================================
bool cudaGEMM(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c,
    int M, int N, int K)
{
    // 计算总字节数
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // ========== 1. 在GPU上分配内存 ==========
    float* dev_A, * dev_B, * dev_C;
    cudaError_t status;

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

    // ========== 3. 配置线程并启动核函数 ==========
    // 每个线程计算C的一个元素
    // 线程块大小：16×16（每个块256个线程）
    dim3 threadsPerBlock(16, 16);

    // 网格大小：向上取整，确保覆盖所有元素
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,  // 列方向
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y   // 行方向
    );

    std::cout << "CUDA配置:" << std::endl;
    std::cout << "  线程块: " << threadsPerBlock.x << "×" << threadsPerBlock.y
        << " = " << threadsPerBlock.x * threadsPerBlock.y << " 个线程" << std::endl;
    std::cout << "  网格: " << blocksPerGrid.x << "×" << blocksPerGrid.y
        << " = " << blocksPerGrid.x * blocksPerGrid.y << " 个块" << std::endl;
    std::cout << "  总线程: " << blocksPerGrid.x * blocksPerGrid.y *
        threadsPerBlock.x * threadsPerBlock.y << std::endl;

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
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
        return false;
    }

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
// 打印矩阵的辅助函数
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
// 主函数
// ============================================
int main() {
    // ========== 定义矩阵大小 ==========
    const int M = 3;  // A的行数 = C的行数
    const int K = 4;  // A的列数 = B的行数
    const int N = 5;  // B的列数 = C的列数

    std::cout << "矩阵维度: " << M << "×" << K << " · "
        << K << "×" << N << " = " << M << "×" << N << std::endl;
    std::cout << std::endl;

    // ========== 使用一维vector模拟二维矩阵 ==========
    std::vector<float> A(M * K);  // M行K列
    std::vector<float> B(K * N);  // K行N列
    std::vector<float> C(M * N);  // M行N列（结果）

    // 初始化A和B
    std::cout << "初始化矩阵..." << std::endl;
    for (int i = 0; i < M * K; i++) {
        A[i] = rand() % 10;  // 0-9的随机数
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = rand() % 10;
    }

    // ========== 串行版本（用于验证结果） ==========
    std::cout << "\n--- 串行计算 ---" << std::endl;
    std::vector<float> C_serial(M * N, 0.0f);

    // 三重循环：串行GEMM
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_serial[i * N + j] = sum;
        }
    }
    std::cout << "串行计算完成！" << std::endl;

    // ========== CUDA版本 ==========
    std::cout << "\n--- CUDA计算 ---" << std::endl;
    bool success = cudaGEMM(A, B, C, M, N, K);

    if (!success) {
        std::cerr << "CUDA计算失败！" << std::endl;
        return 1;
    }
    std::cout << "CUDA计算成功！" << std::endl;

    // ========== 打印矩阵 ==========
    std::cout << std::endl;
    printMatrix(A, M, K, "矩阵A");
    std::cout << std::endl;
    printMatrix(B, K, N, "矩阵B");
    std::cout << std::endl;
    printMatrix(C_serial, M, N, "串行结果C");
    std::cout << std::endl;
    printMatrix(C, M, N, "CUDA结果C");

    // ========== 验证结果是否正确 ==========
    std::cout << std::endl;
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - C_serial[i]) > 1e-5) {
            correct = false;
            std::cout << "❌ 验证失败！位置 " << i
                << ": CUDA=" << C[i]
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

    cudaDeviceReset();
    return 0;
}
