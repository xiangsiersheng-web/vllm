# 先理解这些概念

这页配合 [首次部署教程](README.md) 阅读。命令不必背，先知道它在修改什么、影响多长时间。

## .venv 是什么，应该放在哪里

`.venv` 是一个常用目录名，表示这个项目的 Python 虚拟环境。
前面的点让它在 Linux 中成为隐藏目录；它不是 Git 分支，也不是模型目录。
里面通常有：

```text
.venv/
  bin/python       项目使用的 Python 入口
  bin/activate     激活环境的 shell 脚本
  bin/vllm         安装 vLLM 后生成的命令入口
  lib/...          安装的 Python 依赖
  pyvenv.cfg       解释器版本等环境信息
```

虚拟环境通常仍依赖机器上的基础解释器，它不是一套完整独立操作系统。
它主要隔离 Python 包，防止不同项目的 torch、transformers 等版本互相覆盖。

两种放置方式都可以：

| 方式 | 示例 | 适合情况 |
|---|---|---|
| 项目内 | `/root/vllm-practice/.venv` | 项目与环境一一对应，路径直观，本教程采用 |
| 集中管理 | `/root/venvs/vllm-practice` | 多个环境集中放置，靠绝对路径选择环境 |

放在 `/root/.venv` 本身没有错误，但名字不容易看出属于哪个项目。
如果被多个项目混用，就容易出现依赖互相覆盖；这不是虚拟环境位置本身造成的限制。

### 创建、激活、安装是三件事

- `uv venv .venv`：创建环境目录；相对路径取决于运行时的当前目录。
- `source .venv/bin/activate`：只修改当前 shell 的 PATH 等变量，不创建、不移动目录。
- `uv pip install ...`：向选定的环境安装依赖。

例如，在 `/root` 运行 `uv venv`，默认得到 `/root/.venv`。
进入 `/root/vllm-practice` 后，同样的 `.venv` 则指向 `/root/vllm-practice/.venv`。
`source` 之后再 `cd` 到别的目录，已经激活的环境不会自动改变。

可以这样检查自己正在用哪一个环境：

```bash
pwd
printf '%s\n' "$VIRTUAL_ENV"
command -v python
```

`VIRTUAL_ENV` 为空通常说明当前没有激活 venv；解释器路径比提示符里的环境名字更可靠。
使用已确认的绝对路径，可以绕开激活状态：

```bash
/root/.venv/bin/python --version
/root/vllm-practice/.venv/bin/python --version
```

只执行实际存在的那一个。安装时也可以显式指定环境：

```bash
uv pip install --python /root/.venv/bin/python -r requirements/lint.txt
```

这条假定当前目录为源码根目录，否则清单文件路径也需要写成绝对路径。

### 为什么不直接移动 .venv

环境里的脚本、入口或激活文件可能保存旧绝对路径。直接 `mv` 可能导致它还能部分运行，
但某些命令仍调用旧路径。通常保留旧环境，在目标目录重新创建并按依赖清单安装更稳妥。
源码、模型缓存、编译产物是其他内容，不会因为重建 venv 自动一起搬迁。

## PATH、export、source、.bashrc 的关系

输入 `uv` 时，shell 会依次到 PATH 列出的目录里查找同名程序。
软件文件存在，但目录不在 PATH，仍会显示 command not found。

```bash
export PATH="$HOME/.local/bin:$PATH"
```

意思是：把当前用户的工具目录放到原来的搜索列表前面，保留原来的目录。
`HOME` 在本服务器 root 用户下为 `/root`。这里读取 HOME，没有修改它。
`export` 让当前 shell 及其后续子进程获得这个变量；它不会写入磁盘配置。

`~/.bashrc` 是本机交互式 Bash 的启动配置。把路径设置写进去，新终端就能读取。
`source ~/.bashrc` 则让已经打开的终端马上执行这份配置。
非交互 SSH 可能提前跳过配置，因此自动化命令优先使用工具绝对路径。

uv 的路径全局可见，不代表项目 venv 自动激活：uv 是环境管理工具，venv 是被管理的环境。
不建议在 `.bashrc` 中无条件激活某一个项目；这样每个终端都会被绑定到它。

## Conda base 和项目 venv

`(base)` 表示 Conda 默认环境。`.venv` 是我们为这个项目单独建立的环境。
两者都是管理 Python 和依赖的方式，本次统一使用 uv + `.venv`，减少环境层叠带来的混淆。
`conda deactivate` 只是让当前 shell 退出 base，不会卸载 Conda。
`deactivate` 是退出当前已激活的 venv，也不会删除目录。

## CUDA 三个版本分别是什么

| 名称 | 查看方式 | 本机含义 |
|---|---|---|
| 驱动支持的 CUDA 上限 | `nvidia-smi` | 显示 13.2，表示驱动的兼容能力 |
| 本地 CUDA Toolkit | `/usr/local/cuda/bin/nvcc --version` | 实际编译器为 13.0.88 |
| PyTorch 构建使用的 CUDA | 安装后查看 `torch.version.cuda` | cu130 对应 CUDA 13.0 系列 |

`TORCH_CUDA_ARCH_LIST=12.0` 中的 12.0 是 5090 的计算能力，表示 GPU 架构。
它不表示安装 CUDA Toolkit 12.0。A100 的 8.0 不能照抄到 5090。

## fork、clone、commit、push、pull

| 概念 | 修改的位置 | 干什么 |
|---|---|---|
| fork | GitHub | 创建属于你的上游仓库副本 |
| clone | 本地或云端磁盘 | 把远端 Git 历史和源码下载下来 |
| branch | 当前 Git 仓库 | 给一条实验开发历史起名字 |
| add | 当前仓库的暂存区 | 选择这次提交准备包含哪些内容 |
| commit | 当前 Git 仓库 | 把暂存区保存成一个可追溯的版本 |
| push | GitHub 远端 | 上传已经提交的 Git 历史 |
| pull | 当前 Git 仓库 | 获取并整合远端更新 |

`commit -s` 添加作者对提交的签署声明，不是密码学签名；仅在理解提交内容后使用。
`Co-authored-by: Codex` 记录本次 AI 协助归属。

本地和服务器拥有各自独立的 Git checkout、venv、进程与缓存。
Git 只同步被跟踪并提交的文件，不同步你终端中的激活状态。
分支名只是名字；`git rev-parse HEAD` 给出当前提交 SHA，可用来检查两边是否一致。

`origin`、`fork`、`upstream` 都只是 remote 别名。
当前 Windows 的 origin 指官方、fork 指自己；从个人 fork clone 的云端 origin 会指自己。
所以不能只看别名猜仓库归属，应看 `git remote -v`。

## editable、预编译、CMake 分别做什么

| 操作 | 得到什么 | 改代码后要做什么 |
|---|---|---|
| `uv pip install -e .` | Python 包从源码树加载的开发安装 | `.py` 改动后重启进程 |
| 加 `VLLM_USE_PRECOMPILED=1` | 复用官方 CUDA/Rust 二进制 | 不能据此验证自己的 `.cu` 改动 |
| CMake configure | 根据工具链生成编译计划 | 配置或工具链变了需要重新检查 |
| CMake build + install | 将本机编译的共享库安装回源码树 | 重启服务，加载新产物 |

不是每次改一个文件都要重编整个项目。增量编译会尽量只处理受影响的目标。
ccache 能复用之前相同输入的编译结果，但不保证第一次构建就快。

## 内存、显存、swap、共享内存

GPU 显存放模型权重、KV cache 等 GPU 数据；主机 RAM 供 Python、编译器及系统使用。
`free -h` 看主机 RAM，`nvidia-smi` 看 GPU 显存，不能互相加起来当一种内存。
`available` 比单纯的 `free` 更适合判断还能启动多少新工作。

swap 是磁盘交换空间，内存紧张时可缓冲，频繁换页会明显变慢。
`/dev/shm` 是共享内存文件系统，7.9G 是上限，不是额外 7.9G RAM，也不是提前占满了这些内存。
