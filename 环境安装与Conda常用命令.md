# Linux 与 Conda 常用命令

本文用于课程 Linux 服务器的日常操作。所有文件操作都应限定在自己的 `$HOME` 目录内，不修改系统软件、服务器驱动或他人文件。本文不提供服务器关机、重启或管理员操作命令。

## 1. 认识当前账号和位置

| 目的 | 命令 | 含义 |
|---|---|---|
| 查看当前用户名 | `whoami` | 确认当前登录的是自己的账号 |
| 查看当前目录 | `pwd` | 输出当前工作目录的完整路径 |
| 查看主目录 | `echo "$HOME"` | `$HOME` 代表当前账号自己的主目录 |
| 查看服务器主机名 | `hostname` | 确认当前连接的服务器 |

终端提示符中的 `$` 不是命令的一部分，复制命令时不要把提示符一起复制。

## 2. 查看文件和目录

| 目的 | 命令 | 含义 |
|---|---|---|
| 列出当前目录 | `ls` | 显示非隐藏文件和子目录 |
| 查看详细列表 | `ls -lh` | 显示权限、大小和修改时间 |
| 包括隐藏文件 | `ls -lah` | 隐藏文件的名称以 `.` 开头 |
| 查看文本全文 | `cat README.md` | 适合内容较短的文件 |
| 分页查看文本 | `less README.md` | 方向键翻页，按 `q` 退出 |
| 查看前 20 行 | `head -n 20 README.md` | 快速查看文件开头 |
| 查看后 20 行 | `tail -n 20 README.md` | 快速查看文件末尾 |

Linux 的文件名区分大小写。`README.md` 和 `readme.md` 会被视为两个不同文件。

## 3. 切换目录

| 目的 | 命令 | 含义 |
|---|---|---|
| 返回主目录 | `cd "$HOME"` | 与 `cd ~` 作用相同 |
| 进入课程项目 | `cd "$HOME/2026DeeplearningDemo"` | 路径不同时换成实际项目目录 |
| 进入子目录 | `cd 01_mlp_lab` | 相对于当前目录切换 |
| 返回上一级 | `cd ..` | `..` 代表父目录 |
| 返回上次所在目录 | `cd -` | 在两个目录之间切换 |

路径中有空格时应使用引号，例如 `cd "$HOME/my project"`。

## 4. 安全的文件操作

以下命令仅作为格式示例，执行时将示例文件名换成实际文件名。

| 目的 | 命令 | 安全说明 |
|---|---|---|
| 创建目录 | `mkdir -p "$HOME/practice"` | `-p` 会同时创建缺失的上级目录 |
| 复制文件 | `cp -i source.py backup.py` | `-i` 在覆盖现有文件前询问 |
| 移动或重命名 | `mv -i old_name.py new_name.py` | `-i` 在覆盖前询问 |
| 删除一个文件 | `rm -i unwanted_file.py` | `-i` 要求再次确认；删除后通常不能从回收站恢复 |

执行复制、移动或删除前，先用 `pwd` 确认当前目录，再用 `ls -lah` 确认目标。不要对不确定的路径使用递归强制删除参数。

## 5. 查看磁盘、GPU 和自己的进程

| 目的 | 命令 | 含义 |
|---|---|---|
| 查看主目录总占用 | `du -sh "$HOME"` | 结果可能需要等待一会儿 |
| 查看所在磁盘空间 | `df -h "$HOME"` | 查看已用和可用容量 |
| 查看 GPU 使用情况 | `nvidia-smi` | 显示 GPU、显存和相关进程 |
| 查看当前账号的进程 | `ps -u "$USER" -o pid,etime,cmd` | 仅列出当前用户的进程 |

前台命令运行时，按 `Ctrl+C` 可请求停止当前命令。

## 6. Conda 常用命令

### 6.1 加载和切换环境

| 目的 | 命令 |
|---|---|
| 终端找不到 Conda 时手动加载 | `source "$HOME/miniconda3/etc/profile.d/conda.sh"` |
| 查看 Conda 版本 | `conda --version` |
| 查看全部环境 | `conda env list` |
| 进入课程环境 | `conda activate dl2026` |
| 退出当前环境 | `conda deactivate` |
| 查看当前 Python 位置 | `which python` |
| 查看当前 Python 版本 | `python --version` |

激活成功后，终端提示符前通常出现 `(dl2026)`。

### 6.2 查看和安装依赖

| 目的 | 命令 |
|---|---|
| 查看 Conda 环境中的包 | `conda list` |
| 查看 pip 包 | `python -m pip list` |
| 查看 PyTorch 详细信息 | `python -m pip show torch` |
| 安装项目通用依赖 | `python -m pip install --only-binary=:all: -r requirements.txt` |
| 检查已安装依赖是否冲突 | `python -m pip check` |

PyTorch 的 CUDA 版本按服务器安装笔记单独安装，不写入 `requirements.txt`，避免通用依赖安装时覆盖已经正常工作的 PyTorch。

### 6.3 注册 VS Code Notebook 内核

以下命令只需执行一次：

```bash
# 将当前 dl2026 环境注册为可选的 Notebook 内核。
python -m ipykernel install --user --name dl2026 --display-name "Python (dl2026)"
```

## 7. 每次登录后的最短流程

```bash
# 进入课程环境。
conda activate dl2026

# 进入项目根目录。
cd "$HOME/2026DeeplearningDemo"

# 确认当前位置和 Python 来源。
pwd
which python

# 检查课程环境。
python check_environment.py
```

运行 MLP 快速测试：

```bash
# 从项目根目录运行最短测试。
python 01_mlp_lab/test_mlp_smoke.py
```

使用指定 GPU 时，将编号换成课堂分配的实际编号：

```bash
# 例如仅让当前程序看到编号 0 的 GPU。
CUDA_VISIBLE_DEVICES=0 python 01_mlp_lab/demo.py --device cuda --mode fast --experiment baseline
```

## 8. 查看命令帮助

```bash
# 查看某个命令的简要帮助。
ls --help

# 查看命令手册；按 q 退出。
man ls
```

对命令的作用或目标路径不确定时，先查看帮助，不在共享服务器上试运行系统管理命令。
