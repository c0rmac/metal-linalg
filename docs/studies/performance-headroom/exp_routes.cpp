#include <metal_linalg/core.h>
#include <cstdio>
using namespace metal_linalg;
int main() {
  const char* eb[] = {"cpu","simd","threadgroup","block","tridiag"};
  const char* sb[] = {"cpu","jacobi","block_jacobi","qr_jacobi","qr_block_jacobi","bidiag"};
  const char* qb[] = {"unblocked","streaming_reduced","cpu"};
  std::printf("eigh %s, svd %s, qr %s\n", eigh_policy_source(), svd_policy_source(), qr_policy_source());
  unsigned E[][2] = {{8,4096},{16,4096},{32,256},{32,4096},{64,256},{64,2048},{128,256},{256,64},{512,16},{4096,1}};
  for (auto& c : E) std::printf("eigh %ux%u x%u -> %s\n", c[0], c[0], c[1], eb[(int)eigh_backend(c[0], c[1])]);
  unsigned S[][3] = {{8,8,4096},{32,32,4096},{64,64,256},{128,128,64},{256,256,16},{512,512,4},{1024,64,64},{2048,256,16},{4096,4096,1}};
  for (auto& c : S) std::printf("svd %ux%u x%u -> %s\n", c[0], c[1], c[2], sb[(int)svd_backend(c[0], c[1], c[2])]);
  unsigned Q[][3] = {{16,16,10000},{64,64,1000},{256,128,1000},{512,512,32},{1024,512,16}};
  for (auto& c : Q) std::printf("qr %ux%u x%u -> %s\n", c[0], c[1], c[2], qb[(int)qr_backend(c[0], c[1], c[2])]);
}
