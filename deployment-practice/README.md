# RTX 5090：从空白镜像到自己的 vLLM 服务

更新：2026-11-08。本文用于个人部署学习，当前实践分支是 `test/deploy-practice`。

这份教程按实际进度组织。每个步骤先说明目的，再给命令和验收结果。
不要一次粘贴整篇；一小步确认成功，再往下走。命令失败时先解决当前错误。

## 先看你现在走到哪里

| 步骤 | 状态 | 接下来是否需要执行 |
|---|---|---|
| 创建本地实践分支 | 已完成，基于官方 main 的 `4683d465415b2ef2ec929d44d08197db0993bf89` | 不用重新建 |
| 5090 环境探测 | 已完成 | 机器重建后才需重查 |
| 安装基础工具 | 用户已贴出 uv、git、tmux、CMake、Ninja、ccache 的版本 | 不用重装这些工具 |
| uv 在新终端中可用 | 已配置 `/root/.bashrc` 并验证 | 旧终端执行一次 `source ~/.bashrc` |
| 本地 fork remote | 已配置为 `https://github.com/xiangsiersheng-web/vllm.git` | GitHub 分支发布状态尚未核实 |
| 云端获取个人实践分支 | 已确认 `/root/vllm-practice` 存在且当前为实践分支，未核对两端 SHA | 不重复 clone，检查远端与提交 |
| Python 环境 | 已建在 `/root/.venv`，Python 3.12.3，已含 pre-commit 等工具；项目内还没有 `.venv` | 先按第 3 步统一目录 |
| vLLM 安装、模型服务 | 尚未确认完成 | 环境路径统一后继续 |

先读本页完成首次部署；遇到术语看 [概念与命令解释](CONCEPTS.md)。
服务跑通以后，再看 [调度练习与 CUDA 编译](ADVANCED.md)。

## 0. 两台电脑分别做什么

| 执行位置 | 负责什么 | 怎么确认位置 |
|---|---|---|
| Windows PowerShell | 编辑源码、提交到 Git、推送个人 fork | 路径以 `D:\` 或 `C:\` 开头 |
| 云端 Bash | 安装 Linux 依赖、编译、下载模型、运行服务 | 提示符类似 `root@ubuntu24:~#` |

本地源码目录：`D:\E\工作\ai infra\projects\open-source\vllm`。
云端源码目录统一约定为：`/root/vllm-practice`。

从 Windows 登录服务器：

```powershell
ssh galaxy-vllm-deploy
```

`galaxy-vllm-deploy` 是 Windows SSH 配置中的别名；服务器不一定认识这个别名。
复制命令时不要复制提示符里的 `(base) root@ubuntu24:~#`。
末尾的 `\` 表示命令下一行还没结束，复制时要包含后面的行。

## 1. 理解已经准备好的环境

| 项目 | 已确认的值 | 对这次部署的作用 |
|---|---|---|
| 系统 | Ubuntu 24.04.3，x86_64 | 提供 Linux 软件与编译工具 |
| GPU | RTX 5090，32607 MiB 显存，sm_120 | 存放模型并运行 GPU 运算 |
| 驱动 | 595.84，nvidia-smi 显示 CUDA 13.2 | 让程序与 GPU 通信；13.2 是驱动支持上限 |
| Toolkit | CUDA 13.0，nvcc 13.0.88 | 将 CUDA 源码编译为 GPU 可运行的代码 |
| 编译器 | GCC/G++ 13.3 | 编译项目中的 C/C++ 代码 |
| RAM / swap | 15 GiB / 8 GiB | 编译时使用；与 GPU 显存分开计算 |
| 磁盘 | 根分区约剩 158G | 保存依赖、模型、源码和编译产物 |
| 共享内存 | `/dev/shm` 上限 7.9G | 多进程共享数据，实际使用时才消耗内存资源 |
| 工具 | uv 0.13.7、Git 2.43、tmux 3.4、CMake 3.28.3、Ninja 1.11.1、ccache 4.9.1 | 已准备好，不用重复安装 |

驱动显示 13.2、Toolkit 是 13.0、后面安装 PyTorch cu130，可以同时成立。
详细关系见 [CUDA 三种版本](CONCEPTS.md#cuda-三个版本分别是什么)。

### 新终端找不到 uv 怎么办

uv 位于 `/root/.local/bin/uv`。已经把该目录加入服务器的 `~/.bashrc`：
新交互终端会自动读取；旧终端执行：

```bash
source ~/.bashrc
command -v uv
uv --version
```

验收：输出 `/root/.local/bin/uv` 和版本号。无需重装 uv。
`source` 是让当前终端读取文件；`PATH` 是终端搜索可执行文件的目录列表。

### 仅供机器重建后使用：补工具

当前机器无需重复安装已经验证过的工具。首次安装命令留在这里，供重建时复用：

```bash
apt-get update
apt-get install -y git curl ca-certificates build-essential \
  cmake ninja-build ccache tmux python3.12-dev python3.12-venv
```

`update` 更新可安装软件的清单，不会将系统所有软件升级。
`install` 才安装列出的软件，`-y` 自动回答安装确认。
`python3.12-dev` 提供编译扩展需要的 `Python.h`；`python3.12-venv` 提供系统的虚拟环境支持。
这两个包尚未单独复查安装状态，后续遇到头文件或 venv 错误时优先核查。
当前用户是 root，所以这里不需要 `sudo`。

重建机器安装 uv 的方式见 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。
官方安装器需要访问网络；终端建议的 snap 安装方式不是本教程采用的路线。

## 2. 让本地代码通过自己的 fork 到达云端

### 为什么不直接从官方仓库 clone

本次要保留自己的教程与代码实验，因此采用：

```text
官方 main → 本地 test/deploy-practice → 个人 GitHub fork → 云端相同分支
```

官方 main 是获取更新的来源。自己的 fork 是保存个人提交的地方。
clone 只下载远端已有的提交；本地未提交、未推送的文件不会自动出现到云端。
两台机器的分支名字相同，也不代表内容相同，应比较 commit SHA。

### 2.1 Windows：检查远端和分支

```powershell
cd "D:\E\工作\ai infra\projects\open-source\vllm"
git branch --show-current
git remote -v
git status --short
```

验收：当前分支为 `test/deploy-practice`，`origin` 指向官方仓库，`fork` 指向你的仓库。
这些 remote 名称只是本地起的别名，不是 Git 强制规定的命名。
当前仓库已经加过 `fork`，不要再运行 `git remote add fork ...`。

若 GitHub 上还没创建 fork：打开 [官方仓库](https://github.com/vllm-project/vllm)，点击 Fork，
Owner 选择自己，保留仓库名 vllm，勾选只复制 main，再点击 Create fork。
本地配置了地址不等于 GitHub 上的仓库一定已经创建成功。

### 2.2 Windows：审阅、提交、推送

先阅读本次新增的 `deployment-practice/` 文件。`git status --short` 中 `??` 表示文件尚未纳入 Git。
确认内容后再提交；Git 只会把加入暂存区的内容放入这次提交。

```powershell
git add deployment-practice/
git diff --cached --stat
git diff --cached
```

如果输出包含你不打算提交的其他内容，先处理暂存区，再继续。
仓库要求提交带人类签署与 AI 协助归属；下面的 `-s` 会用 Git 中配置的姓名、邮箱生成签署行：

```powershell
git config user.name
git config user.email
git commit -s -m "docs: explain RTX 5090 deployment practice" -m "Co-authored-by: Codex"
git push -u fork test/deploy-practice
```

若姓名、邮箱为空，先用 `git config user.name "你的姓名"` 和
`git config user.email "你的邮箱"` 配置当前仓库，再提交。
本教程在个人 fork 中保存，不需要向官方提 PR。

`push -u fork test/deploy-practice`：把该分支提交上传到 fork，并记住默认跟踪目标。
它不会上传未提交文件、模型缓存或 `.venv`。
如果已有相同提交，则无需重复 commit，直接推送即可。

**代理要根据当前电脑实际状态选择。** 之前记录的 7897 在本轮本地探测时连接失败：
这表示代理没有在该端口监听，不代表 GitHub 仓库不存在。
可在 Windows 查看代理软件实际端口，并执行 `Test-NetConnection 127.0.0.1 -Port 7897` 检查。
如果当前无需代理，可仅对这一次推送关闭 Git 配置中的代理：

```powershell
git -c http.proxy= -c https.proxy= push -u fork test/deploy-practice
```

若确认本机代理已在 7897 监听，则使用：

```powershell
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push -u fork test/deploy-practice
```

`-c` 仅影响这一条命令，不会永久修改 Git 配置。
`127.0.0.1` 指当前电脑；Windows 的代理地址不能直接搬到服务器使用。
GitHub 若要求认证，按本机凭据管理器提示登录。

### 2.3 云端：首次下载自己的分支

先确认 `/root/vllm-practice` 不存在。如果之前已经 clone 过，不要重复 clone 或删除目录，走下方已有目录分支。

```bash
cd /root
git clone --branch test/deploy-practice \
  https://github.com/xiangsiersheng-web/vllm.git vllm-practice
cd /root/vllm-practice
git branch --show-current
git rev-parse HEAD
```

`--branch` 指定下载后切到哪个分支；最后的 `vllm-practice` 是本机目录名，可以与 GitHub 仓库名不同。
保留正常 Git 历史，便于后续判断预编译产物的来源。

如果目录已经存在，先查看它的状态：

```bash
cd /root/vllm-practice
git status --short
git remote -v
git branch --show-current
```

若已有本地修改，先保存，不能直接覆盖。
如果云端原先 clone 的是官方仓库，且尚未添加名为 fork 的远端，可执行一次：

```bash
git remote add fork https://github.com/xiangsiersheng-web/vllm.git
git fetch fork
git switch test/deploy-practice
git merge --ff-only fork/test/deploy-practice
git branch --set-upstream-to=fork/test/deploy-practice
```

这组命令假定云端已有之前创建的同名分支；如果没有，应使用
`git switch --track fork/test/deploy-practice` 创建跟踪分支。
`--ff-only` 只允许沿历史直接前进，有分叉时会停下，便于先查清差异。

**验收：** 本地与云端的 `git rev-parse HEAD` 相同，而且都能看到已提交的教程。
本次官方源码基线是 `4683d465...`；增加教程提交以后 HEAD 会变化，这是正常的。
不要再把 HEAD 强行切回旧 SHA，否则新教程提交就不在当前分支顶端了。

## 3. 云端：为这个项目创建独立 Python 环境

### 先处理这台机器目前的环境位置

本轮只读检查发现：你在 `/root` 创建了 `/root/.venv`，里面已有 pip、pre-commit 等工具；
`/root/vllm-practice/.venv` 还不存在。旧环境可以使用，不是创建失败。
但是教程约定进入项目后用 `.venv/bin/python`，因此统一放在项目内会更容易理解和复现。

`source .venv/bin/activate` 只激活环境，不创建目录。真正决定创建位置的是 `uv venv` 的执行目录或路径参数。
详情见 [什么是 .venv](CONCEPTS.md#venv-是什么应该放在哪里)。

这次建议保留 `/root/.venv`，在项目内另建。先退出当前激活的 venv（仅当已激活时执行）：

```bash
printf '%s\n' "$VIRTUAL_ENV"
```

如果显示 `/root/.venv`，执行 `deactivate`。若退出后仍显示 `(base)`，再执行 `conda deactivate`。
然后只执行下面这组明确路径的命令：

```bash
cd /root/vllm-practice
pwd
uv venv --python /usr/bin/python3.12 --seed /root/vllm-practice/.venv
source /root/vllm-practice/.venv/bin/activate
.venv/bin/python -c 'import sys; print(sys.executable)'
```

预期路径为 `/root/vllm-practice/.venv/bin/python`。然后在新环境重做第 3.3 步安装检查工具，
并重新运行 `pre-commit install`，避免 hook 仍指向旧环境。
不用删除旧环境，也不要直接移动它；已创建正确环境后，就不再重复执行第 3.1 的创建命令。
如果旧环境以后已装入 torch/vLLM 等大量依赖，先记录版本与安装方式，再决定是否重建。

### 3.1 为什么要退出 Conda base

终端的 `(base)` 表示 Conda 默认环境已激活，它的 Python 为 3.14.7。
当前源码允许这个 Python 范围，但本教程选用 3.12，减少第三方依赖 wheel 不齐全的变量。
不需要删除 Conda，也不需要把 base 的 Python 降级。

```bash
cd /root/vllm-practice
conda deactivate
uv venv --python /usr/bin/python3.12 --seed
source .venv/bin/activate
```

若终端没有启用 Conda，跳过 `conda deactivate`。
若 `.venv` 已存在，先检查解释器版本，不重复创建或删除它。

| 命令 | 为什么要做 |
|---|---|
| `uv venv` | 为此项目建立独立的 Python 和依赖安装位置 |
| `--python /usr/bin/python3.12` | 明确选系统 3.12，避免 uv 选到 Conda 的 3.14 |
| `--seed` | 在环境里提供 pip 等种子工具；后续安装仍统一使用 uv |
| `source .venv/bin/activate` | 让当前终端优先使用这个项目的工具 |

验收：

```bash
.venv/bin/python --version
.venv/bin/python -c 'import sys; print(sys.executable)'
```

应显示 Python 3.12.x 和 `/root/vllm-practice/.venv/bin/python`。
创建 venv 只做一次；每个新终端需要重新激活，或者直接调用 `.venv/bin/...`。

### 3.2 设置 CUDA 与编译参数

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=2
export NVCC_THREADS=1
nvcc --version
```

| 设置 | 作用 |
|---|---|
| `CUDA_HOME` | 告诉构建工具去哪里找 CUDA Toolkit |
| 把 `$CUDA_HOME/bin` 加入 PATH | 让终端能直接找到 nvcc |
| `TORCH_CUDA_ARCH_LIST=12.0` | 为 5090 架构编译；这是计算能力，不是 CUDA 版本 |
| `MAX_JOBS=2` | 限制部分构建流程并行任务数，减轻 15 GiB RAM 的压力 |
| `NVCC_THREADS=1` | 限制 nvcc 的内部线程并行度；不等于只用一个 GPU 线程 |

预期 nvcc 显示 release 13.0。以上 export 对当前终端有效；进阶 CMake 还要单独限并行。

### 3.3 安装开发检查工具

```bash
uv pip install -r requirements/lint.txt
.venv/bin/pre-commit install
```

`-r` 让 uv 读取仓库维护的依赖清单。这里安装代码格式等检查工具。
`pre-commit install` 将检查钩子安装到这个 Git checkout，后续提交时帮助发现格式问题。
首次运行检查钩子可能还需下载工具；服务器尚未安装 vLLM 是正常的。

## 4. 云端：先安装可运行的 vLLM 基线

先跑通服务，再学习编译，能帮助区分模型/依赖问题和自己修改代码带来的问题。
这一步复用官方二进制，不代表已经完成本机 CUDA 编译。

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
  uv pip install -e . --torch-backend=cu130
```

| 参数 | 含义 |
|---|---|
| 命令前的两个变量赋值 | 仅对紧随其后的安装命令生效 |
| `VLLM_USE_PRECOMPILED=1` | 复用官方构建的 CUDA/Rust 产物，先跳过本机编译 |
| `VLLM_PRECOMPILED_WHEEL_VARIANT=cu130` | 选择官方 CUDA 13.0 产物系列 |
| `uv pip install` | 将软件及依赖安装到已激活的 venv |
| `-e .` | editable 安装当前目录；Python 代码从这个源码树加载 |
| `--torch-backend=cu130` | 从对应 CUDA 13.0 的 PyTorch 软件源解析依赖 |

当前源码锁定 `torch==2.14.0`，由依赖文件决定，不要再照旧 0.32.0 笔记强装 2.13。
wheel 是打包好的分发文件，里面可以包含 Python 代码和已编译的二进制。
固定源码基线对应的 cu130 元数据已探测到 200，完整安装仍需实际验证。
若安装器报告无匹配产物，保留错误日志，不强行指定其他 nightly 绕过兼容检查。

安装成功后继续：

```bash
.venv/bin/vllm download-kernels
.venv/bin/python -c 'import vllm, torch; from vllm import _C; print("vllm:", vllm.__version__); print("source:", vllm.__file__); print("torch:", torch.__version__); print("torch CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0)); print("extension:", _C.__file__); print(torch.ones(1, device="cuda"))'
```

`download-kernels` 提前下载所需 kernel 缓存，减少第一次请求时的临时下载。
后面的检查确认源码来源、PyTorch 版本、GPU 可见性和一次真实 GPU 张量运算。
成功标准：没有异常；source 指向本项目，GPU 为 RTX 5090，CUDA 张量打印成功。
修改 `.py` 后需要重启服务；editable 不会刷新已运行的进程。修改 `.cu` 必须重新编译。

## 5. 云端：在 tmux 中启动模型服务

tmux 可以让程序在 SSH 断开后继续运行。先创建一个会话：

```bash
tmux new -s vllm-serve
```

进入 tmux 相当于进入另一个终端。环境不能凭提示符猜测，明确设置：

```bash
cd /root/vllm-practice
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p /root/vllm-practice-results
set -o pipefail
.venv/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 \
  --max-num-seqs 8 --gpu-memory-utilization 0.80 \
  2>&1 | tee /root/vllm-practice-results/serve-baseline.log
```

| 参数 | 为什么先这样设 |
|---|---|
| 模型 ID | 使用 Qwen 的 4B non-thinking 文本模型，先减少模型复杂度 |
| `127.0.0.1` | 服务先只监听服务器本机，Windows 通过隧道访问 |
| `8000` | 服务监听的 TCP 端口，后面测试要使用同一端口 |
| `bfloat16` | 以 BF16 精度运行，作为未量化基线 |
| `8192` | 限制单请求输入与输出的总上下文，先验证 8K 再扩到 32K |
| `--max-num-seqs 8` | 限制同时处理的序列数量，从小并发开始 |
| `0.80` | vLLM 的 GPU 内存预算比例，不代表权重恰好占 80% |
| `tee` | 同时把日志显示在屏幕并保存文件 |
| `2>&1` | 把错误输出也并入日志 |
| `pipefail` | 管道中 vLLM 失败时，不被 tee 的成功退出状态掩盖 |

首次启动会下载模型、准备 kernel 并初始化执行图，耗时取决于网络与机器。
等到服务显示就绪再测试，不要只凭进程还在运行判断成功。
模型缓存默认不在 Git 里；记录下载后的模型 revision 便于复现。

tmux 的三个操作：

- 保留服务并离开：先按 `Ctrl+B`，松开后按 `D`。
- 回到服务界面：`tmux attach -t vllm-serve`。
- 停止服务：回到服务界面后按 `Ctrl+C`。

不要重复运行启动命令占用同一个 GPU 或端口。如果提示 session 已存在，先 attach 检查。

## 6. 确认服务能用

在服务器另开一个 SSH 终端，执行：

```bash
curl -i http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

第一条期望 HTTP 200，响应正文可以为空。第二条应返回包含模型 ID 的 JSON。
`curl` 用来发 HTTP 请求；这里的地址是服务器上的本机服务。
`Connection refused` 通常表示服务尚未监听或已经退出，应先看启动日志。

Windows 如果要访问这个服务，在 Windows PowerShell 保持下面的连接打开：

```powershell
ssh -N -L 18000:127.0.0.1:8000 galaxy-vllm-deploy
```

`-L` 把 Windows 的 18000 端口，经 SSH 转到服务器的 8000；`-N` 只建隧道，不打开远程命令行。
然后在 Windows 浏览器打开 `http://127.0.0.1:18000/v1/models`。
本地端口用 18000，是为了避免和 Windows 上已有的 8000 服务冲突。

真实生成请求可先用下一步的小规模 benchmark 验证；它调用现有服务端接口。

## 7. 保存一份可对比的服务基线

先理解指标：TTFT 是收到第一个 token 的等待时间，TPOT 是后续每个输出 token 的平均耗时；
吞吐反映单位时间能完成多少请求或 token。随机输入基准用来测运行行为，不用于评价回答质量。

在服务器第二个终端执行，保持模型服务运行：

```bash
cd /root/vllm-practice
source .venv/bin/activate
.venv/bin/vllm bench serve --help
```

下面是一份固定的小负载。先执行一次作预热，之后重复三次，修改结果文件名为 run2、run3：

```bash
.venv/bin/vllm bench serve \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random --random-input-len 128 --random-output-len 64 \
  --num-prompts 20 --request-rate 2 --max-concurrency 4 --seed 0 \
  --save-result --result-dir /root/vllm-practice-results \
  --result-filename precompiled-run1.json
```

含义：20 个请求，目标每秒到达 2 个，最多 4 个并发，随机输入长度参数 128、输出长度参数 64。
`seed 0` 固定随机种子；实际请求长度和输出长度以工具记录为准。
保存 JSON 后检查失败请求数，不能只看速度。
以后对比修改前后，使用相同模型、命令、服务配置与预热方式。
完整的语义生成、流式与排队观测验收在实际进入对应练习时补齐，不能把健康检查等同于这些测试全部通过。

## 8. 每次重新登录后，应该做什么

进入服务器并准备项目终端：

```bash
cd /root/vllm-practice
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
export NVCC_THREADS=1
```

不需要重复 apt 安装、clone、uv venv 或重装 vLLM。
已启动的服务先用 `tmux ls` 检查；新终端激活 venv 不会替换已运行服务的环境。
uv 的 PATH 已持久化，项目 venv 则按项目需要手动激活，两者分工不同。

代码同步方向约定为 Windows 编辑并提交、云端拉取并运行：

```text
Windows: git add → git commit → git push
云端:    git status --short → 确认无待保存改动 → git pull --ff-only
```

Git 不同步正在运行的进程、虚拟环境或模型缓存。
更新 Python 代码要重启服务；更新 CUDA 代码需要重新编译再重启。
换源码基线或依赖版本后，重新核对二进制是否匹配，不能只做 git pull。
后续还会在 [进阶教程](ADVANCED.md) 中安排日志练习与 CUDA 编译。

## 9. 常见问题先查这里

| 现象 | 先检查什么 |
|---|---|
| `^[[200~command: command not found` | 粘贴混入控制字符，Ctrl+C 后重新粘贴；command 是 Bash 内建命令 |
| `uv: command not found` | `source ~/.bashrc`；再用 `/root/.local/bin/uv --version` 区分 PATH 与安装问题 |
| 仍显示 `(base)` 或 Python 3.14 | 检查是否退出 Conda，并激活项目 venv；以 `sys.executable` 为准 |
| `destination path ... already exists` | 进入现有目录检查状态，不重复 clone 或删除 |
| `Remote branch ... not found` | 本地分支还没推送到 fork，或远端地址写错 |
| Windows 无法连接 127.0.0.1:7897 | 当前代理未监听该端口；检查代理软件，或按第 2 步临时关闭代理 |
| `nvcc: command not found` | 用 `/usr/local/cuda/bin/nvcc --version` 检查，再补 PATH |
| `Python.h: No such file` | 系统 Python 开发头文件缺失；核对 python3.12-dev |
| 安装找不到匹配 wheel | 检查 Python、平台、cu130 和依赖错误，保留完整安装日志 |
| 编译进程 `Killed` | 检查 RAM、swap，停止模型服务，并行任务从 2 降到 1 |
| 端口 8000 已占用 | 检查 tmux 是否已有服务，避免重复启动 |
| `.py` 改了没生效 | 检查实际源码路径，并重启服务及 worker |
| `.cu` 改了没生效 | 先编译安装，再确认产物与加载路径，最后重启服务 |

## 依据与范围

本教程固定上述源码基线，依赖事实以该 checkout 的 `requirements/cuda.txt`、
`requirements/build/cuda.txt`、`pyproject.toml` 为准；不套用旧 notes 中的 v0.32.0 组合。
服务器配置已探测，安装结果按本页状态表记录；后续命令是教程，尚未替用户执行整套部署。

- [vLLM GPU 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM 增量编译工作流](https://docs.vllm.ai/en/latest/contributing/incremental_build/)
- [GitHub Fork 操作说明](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo)
- [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)
- [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
