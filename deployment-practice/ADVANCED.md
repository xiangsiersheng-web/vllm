# 进阶：观察调度、编译 CUDA、验证 kernel 修改

先完成 [首次部署教程](README.md)，确认服务能处理真实生成请求。
本页是后续操作说明，目前尚未执行这些实验。

## 1. 为什么分成三个基线

| 基线 | 源码与二进制状态 | 用途 |
|---|---|---|
| A 预编译基线 | Python 来自源码，native 产物来自官方 | 确认模型、依赖与服务链路可用 |
| B 本机编译基线 | 不改 kernel，使用本机编译的 CUDA 产物 | 隔离工具链与编译选项的影响 |
| C 修改版 | 在 B 上只改一处 kernel 参数再重编 | 判断这处代码变化的作用 |

直接拿 A 与 C 比较，会同时包含代码和编译环境差异，不容易解释结果。
性能数字可以变好、变差或没有显著变化；学习目标是能用证据说明原因。

## 2. 练习一：用日志观察调度

调度器决定哪些请求先进入执行，以及每一步给请求安排多少 token。
候选位置为 `vllm/v1/core/sched/scheduler.py` 的 `schedule()`。
先读当前调用链，确认实际采用的 scheduler 类及异步调度配置，再决定插入位置。

建议只记录运行/等待请求数、当步调度 token 数、新接纳请求。
日志应可关闭或节流；每个 decode step 都写 INFO 会干扰性能，还会迅速占用磁盘。
本页不提前给未经运行验证的 patch，实际练习时按当前源码编写、检查。

操作顺序：

1. 保存 A 的启动配置与结果。
2. 添加最小日志改动，重启服务。editable 只保证新进程读取新源码。
3. 用 `--max-num-seqs 2` 和四个足够长的并发请求制造排队。
4. 观察出现 waiting、请求逐步进入 running，最终全部完成。
5. 关闭日志后再测性能，避免把日志 I/O 算作调度或 kernel 开销。

不要求日志严格出现 `waiting=4 → 2 → 0`，到达和完成时序会影响采样。
测试依赖按仓库要求安装，使用现有 scheduler 测试：

```bash
cd /root/vllm-practice
source .venv/bin/activate
uv pip install -r requirements/test/cuda.in --torch-backend=cu130
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py -x -q
```

测试依赖可能较多；安装前后核对 torch 版本，不将依赖变化混入性能对比。
`-x` 表示第一次失败就停止，`-q` 减少输出。测试失败时先读原因，不直接忽略。
服务层的并发观测和这些单元测试各自验证不同部分，都应保存结果。

## 3. 本机 CUDA 编译：先理解工具分工

| 工具 | 做什么 |
|---|---|
| GCC/G++ | 编译主机侧 C/C++ 代码 |
| nvcc | 编译 CUDA 代码，并协调主机编译器 |
| CMake | 找到依赖、决定编译配置并生成构建计划 |
| Ninja | 按构建计划运行实际编译任务 |
| ccache | 缓存相同编译输入的结果，加快后续重编 |

第一次编译需要处理大量代码，可能数小时；后续相同环境下只改一处代码通常可增量构建。
15 GiB RAM 是主要约束，开始前停止模型服务，在 tmux 中操作，避免 SSH 断开终止构建。

```bash
tmux new -s vllm-build
```

进入会话后：

```bash
cd /root/vllm-practice
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
export NVCC_THREADS=1
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
.venv/bin/python tools/generate_cmake_presets.py
```

构建清单安装的是编译工具及依赖，不能假设之前的运行环境已经包含全部构建依赖。
生成脚本会写 `CMakeUserPresets.json`，保存本机路径与构建参数。
如果文件已经存在，先阅读它，不直接加 force 参数覆盖已有配置。

### 3.1 生成后必须检查 preset

| 配置 | 本次要求 | 原因 |
|---|---|---|
| `VLLM_PYTHON_EXECUTABLE` | `/root/vllm-practice/.venv/bin/python` | 必须与安装和运行 vLLM 的环境一致 |
| `CMAKE_CUDA_COMPILER` | `/usr/local/cuda/bin/nvcc` 或其真实路径 | 指定已确认的 CUDA 13.0 编译器 |
| `CMAKE_INSTALL_PREFIX` | 源码根目录，如生成的 `${sourceDir}` | 让新共享库落回 editable 安装加载的位置 |
| `NVCC_THREADS` | `"1"` | 限制 nvcc 内部并行 |
| `CMAKE_JOB_POOLS` | `"compile=2"` | 限制 Ninja 的编译任务池 |
| build preset 的 `jobs` | `2` | 避免按 CPU 核数默认生成过高并行 |

`MAX_JOBS=2` 不会自动覆盖生成脚本按 CPU 数量生成的全部 preset 值。
内存紧张时将相关 job 限制一起降为 1，不只修改 shell 变量。
`TORCH_CUDA_ARCH_LIST` 要在 configure 前设置，检查日志中目标架构是否为 12.0。

### 3.2 配置、构建、安装

```bash
mkdir -p /root/vllm-practice-results
set -o pipefail
cmake --preset release 2>&1 | tee /root/vllm-practice-results/configure.log
```

configure 成功后才继续：

```bash
cmake --build --preset release --target install --parallel 2 \
  2>&1 | tee /root/vllm-practice-results/build.log
```

`--preset release` 使用刚检查的配置。`--target install` 会构建并将产物安装到配置的前缀。
这里的 install 是安装本机编译产物，不是在重新下载 PyPI 里的 vLLM。

成功后检查：

```bash
.venv/bin/python -c 'from vllm import _C; print(_C.__file__)'
stat vllm/_C.abi3.so
sha256sum vllm/_C.abi3.so
```

单看路径不够：预编译产物也可能位于同一位置。
结合构建成功日志、安装记录、产物时间与校验值，确认本次产物已安装；之后重启服务。
再次跑 README 中相同的健康检查和 benchmark，文件名改为 `local-build-run1.json` 等，保存 B。

CMake 主要构建 C++/CUDA 组件；预编译阶段复用的 Rust 前端并没有因此自动全部重编。
若以后需要全 native 源码构建，再按根目录 `rust-toolchain.toml` 和 Rust 构建说明处理。
当前不修改 Rust 代码。

## 4. 练习二：改变一次真实 kernel launch 参数

候选位置：`csrc/layernorm_kernels.cu` 中 `rms_norm` 的 `max_block_size`。
原表达式为：

```cpp
batch_invariant_launch ? 1024 : ((num_tokens < 256) ? 1024 : 256);
```

计划仅将最后一个 256 改为 128。它改变的是非 batch-invariant、`num_tokens >= 256` 时的上限。
因此 batch=1 的短 decode 不会命中这个改动；真实 block 大小还受 hidden size、dtype、向量化限制。
block 大小是一次 GPU launch 中每个线程块的线程数，与 `MAX_JOBS` 的主机编译任务数是两回事。

### 4.1 先定义验证目标，再改代码

- 模块用途：执行 RMSNorm 数值归一化。
- 输入输出：输入 tensor、weight、epsilon，产生归一化 tensor。
- 防止失败：归约错误、越界、数值回归。
- 最便宜的验证：先复用现有 layernorm 测试，再验证真实服务。

先修改、重编安装，然后运行：

```bash
.venv/bin/python -m pytest tests/kernels/core/test_layernorm.py -x -q
```

还需确认现有参数覆盖改动阈值两侧、BF16 和目标 hidden size；有缺口时再按测试指南补最小用例。
不能只因命令退出成功就假设目标分支一定执行过。

### 4.2 先看算子，再看整个服务

复用 `benchmarks/kernels/benchmark_layernorm.py`，不在 tests 里另加一次性 benchmark。
实际测量前先确认该脚本的 RMSNorm 调用是否命中被修改的 CUDA 实现，可能存在融合或其他后端路径。
建议形状包括 tokens=1/128/256/512、hidden_size=2560、BF16，固定预热与重复方法。
这部分只规定实验设计，具体计时与 dispatch 核查在执行实验时完成。

对比 B 与 C 的 kernel 数据，再用相同服务负载作端到端对比，分别报告。
`bench latency` 和 `bench serve` 测量对象不同，不能直接把两者结果当作前后对照。
服务若没有调用目标 kernel，端到端性能不变也不能证明改动没有作用。
真实输出有变化时，需要分析并补模型评估，不以微基准替代正确性验证。

## 5. 保存和恢复

每阶段保存源码 SHA/diff、模型 revision、依赖版本、环境变量、启动命令、构建与测试日志。
结果放 `/root/vllm-practice-results`；不要将 `.venv`、大权重、缓存和二进制提交到 Git。

恢复顺序：停止服务 → 查看 git diff → 只撤回本次实验改动 → 重编安装 → 重启服务 → 重跑验收。
如果 diff 中混有其他工作，先拆分保存，不用整文件恢复覆盖别人的改动。
保留 A、B、C 三组结果，出现错误时才有可比较的证据。
