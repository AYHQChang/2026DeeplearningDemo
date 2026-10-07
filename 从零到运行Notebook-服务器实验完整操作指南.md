# 从零到运行 Notebook：服务器实验完整操作指南

本文件是课程服务器操作的统一入口，供学生首次配置、日常上课和教师维护时查阅。适用流程：**Windows 上的 VS Code → SSH 连接 Linux 课程服务器 → 在个人目录运行课程 Notebook**。

课程实验内容、模型组件和作业要求分别见各实验室的 README。本指南集中说明连接、环境、代码副本、运行、更新和故障恢复。整理日期：2026-10-07；服务器地址、端口、账号及 GPU 分配以教师当堂通知为准。

| 现在要做什么 | 从哪里开始 |
|---|---|
| 第一次使用服务器 | [1. 准备与路径](#start) → [2. 连接服务器](#connect) → 按顺序完成第 3–6 节 |
| 已经配好环境，开始上课 | [7. 日常运行与安全更新](#student-update) |
| 安装时报空间不足 | [8. 清理空间与删除多余环境](#recovery) |
| 教师首次部署、发布新代码 | [9. 教师维护与发布](#teacher) |
| SSH、模块导入、内核报错 | [10. 故障定位](#troubleshooting) |
| 查 Linux / Conda 常用命令 | [11. 命令速查与完成标准](#reference) |

<a id="start"></a>
## 1. 准备与路径：先分清在哪台电脑操作

Windows 电脑负责连接和显示界面，Python 训练在 Linux 服务器执行。这条路线只需要在 Windows 准备 VS Code 和 SSH 客户端；Conda、Python、PyTorch 安装在服务器个人账号中。Windows 本地训练属于另一条路线，历史资料见[归档索引](./docs/archive/2026-10-07/README.md)。

### 1.1 课堂需要提供的信息

| 信息 | 本文占位写法 | 说明 |
|---|---|---|
| 服务器地址 | `xxx.xxx.xxx.xxx` | 替换为课堂提供的地址 |
| SSH 端口 | `xxxx` | 替换为实际数字端口 |
| 自己的账号 | `YOUR_ACCOUNT` | 每人使用自己的账号 |
| 公共课程仓库 | `/data/public/xxx/course-repos/Deeplearning2026` | `xxx` 替换为教师公布的目录名 |
| GPU 编号 | `cuda:N` | 仅在教师分配后使用，默认 CPU |

登录密码只在登录提示中输入，不写进 SSH 配置、Notebook 或 Git 仓库。

### 1.2 四种操作位置

| 本文标注 | 应在哪里执行 |
|---|---|
| **Windows PowerShell** | 本机 PowerShell，尚未登录服务器的终端 |
| **Linux 终端** | SSH 连接成功后，VS Code 新建的远程终端 |
| **Notebook 单元格** | `.ipynb` 中的 Python 代码单元格 |
| **配置文件** | 编辑保存文件，不能当成终端命令执行 |

命令块中的命令按顺序执行；某一步报错先解决，再执行后续步骤。不要复制终端提示符，如 `(dl2026) [账号@主机]$`。

### 1.3 目录约定

```text
GitHub：AYHQChang/2026DeeplearningDemo（教师维护的课程源代码）
  ↓ 教师同步
教师服务器工作副本：$HOME/2026DeeplearningDemo
  ↓ 教师发布
服务器公共 Git 仓库：/data/public/xxx/course-repos/Deeplearning2026
  ↓ 学生克隆 / 更新
学生个人工作副本：$HOME/courses/Deeplearning2026
  ├── 01_mlp_lab
  ├── 02_cnn_lab
  ├── 03_rnn_lab
  └── 04_transformer_lab
```

`$HOME` 会自动展开为当前账号的主目录，不要把它改成其他人的路径。教师与学生的 `$HOME` 不同。公共仓库只用于分发 Git 版本；学生在自己的工作副本里运行、保存作业。

<a id="connect"></a>
## 2. 用 VS Code 连接课程服务器

### 2.1 准备兼容的 VS Code

现有课堂配置针对 CentOS 7 / glibc 2.17，沿用课程提供的 **VS Code 1.98.2** 兼容包。该版本是课程旧服务器的兼容安排。当前 VS Code 官方远程 Linux 要求包含 glibc ≥ 2.28，CentOS 7 不在其支持范围内；服务器升级后由教师重新核对客户端版本。[官方系统要求](https://code.visualstudio.com/docs/remote/linux)

使用课堂 ZIP 包 `VSCode-win32-x64-1.98.2.zip` 时：

1. 解压到自己的可写目录，例如 `D:\DLTools\VSCode-1.98.2`，不要直接在压缩包中运行。
2. 在 `Code.exe` 同一级新建 `data` 文件夹，使用独立的便携配置。
3. 启动这个目录中的 `Code.exe`，在“帮助 → 关于”核对版本。
4. 在本机扩展面板安装课程兼容版本的 **Remote - SSH**。若扩展提示不支持当前 VS Code，使用课堂提供的版本或扩展菜单“安装另一个版本”。
5. 课堂使用期间保持这一套客户端和扩展版本；如需设置更新方式，从设置中搜索 `update mode`、`extensions auto update`，避免兼容客户端被自动替换。

个人电脑原有的其他 VS Code 可以保留；连接本课程服务器时使用上述兼容入口。实验室电脑已配置好时，不必重复安装。

在 **Windows PowerShell** 检查 SSH：

```powershell
ssh -V
```

应显示 OpenSSH 版本。如果提示找不到 `ssh`，先检查 Windows“可选功能”中的 OpenSSH 客户端；受管理的实验室电脑请教师协助处理。

### 2.2 写入自己的 SSH 配置

VS Code 按 `F1`，选择 `Remote-SSH: Open SSH Configuration File...`，选择当前 Windows 用户目录下的 `.ssh/config`。在**配置文件**中填写：

```sshconfig
Host course-server
    HostName xxx.xxx.xxx.xxx
    User YOUR_ACCOUNT
    Port xxxx
    PasswordAuthentication yes
```

替换地址、账号、端口后保存。文件名是 `config`，不是 `config.txt`；`HostName` 只填地址，端口单独写在 `Port`。`course-server` 是自定义连接别名。

在 **Windows PowerShell** 检查实际读取的配置：

```powershell
ssh -G -F "$env:USERPROFILE\.ssh\config" course-server | Select-String '^(hostname|user|port|passwordauthentication) '
```

核对输出中的地址、账号、端口。若使用了其他位置的配置文件，把 `-F` 后的路径换成实际文件。

### 2.3 先验证 SSH，再打开远程窗口

在 **Windows PowerShell**：

```powershell
ssh -F "$env:USERPROFILE\.ssh\config" course-server
```

第一次连接会显示主机指纹，和教师提供的信息核对后确认。输入密码时通常不显示字符或星号，这是正常现象。登录成功后用 `exit` 返回本机。

然后在 VS Code：

1. `F1` → `Remote-SSH: Connect to Host...` → `course-server`。
2. 若询问服务器系统，选择 **Linux**；按提示输入密码。
3. 等待远程窗口打开，左下角应显示 `SSH: course-server`。
4. 选择“终端 → 新建终端”。

在这个 **Linux 终端**检查：

```bash
hostname
whoami
echo "$HOME"
pwd
git --version
```

应看到服务器主机名、自己的 Linux 账号和个人路径。若仍显示 `PS C:\...>`，当前是 Windows 终端。

连接后，在扩展面板的 **SSH: course-server** 一侧安装或启用 **Python** 和 **Jupyter** 扩展。只在本机安装扩展不等于远端已经具备运行 Notebook 的能力；旧 VS Code 使用课堂兼容扩展版本。

### 2.4 连接失败时先看这一段

在 **Windows PowerShell** 检查网络与端口，执行前替换地址和端口：

```powershell
Test-NetConnection xxx.xxx.xxx.xxx -Port xxxx
ssh -vvv -F "$env:USERPROFILE\.ssh\config" course-server
```

`TcpTestSucceeded: True` 只证明端口可达，不能证明账号密码正确。超时先检查校园网 / VPN、地址和端口；`Permission denied` 检查账号、密码和账号是否可用。

若命令行 SSH 能登录，VS Code 却停住或看不到密码提示：`F1` → `Preferences: Open User Settings (JSON)`，把下面字段合并到已有设置中，不覆盖其他设置；不要重复外层大括号。

```json
{
    "remote.SSH.showLoginTerminal": true,
    "remote.SSH.useLocalServer": false,
    "remote.SSH.connectTimeout": 60,
    "remote.SSH.remotePlatform": {
        "course-server": "linux"
    }
}
```

保存后执行 `Developer: Reload Window` 再连接。这组登录提示设置也见 [Remote-SSH 官方排错说明](https://code.visualstudio.com/docs/remote/troubleshooting)。

如果日志已经显示认证成功、传输及解压成功，随后出现 `LinuxPrereqs`、`GLIBC` 或 `exitCode==207`，应核对 VS Code 与服务器系统兼容性。不要因为最后一句“连接失败”反复改密码，也不要自行替换服务器系统库。其他错误从“输出 → Remote - SSH”查看报错附近的日志，反馈时遮住账号和地址。

<a id="conda"></a>
## 3. 检查或安装服务器端 Miniconda

以下全部在 **Linux 终端**，使用自己的账号。已有可用 Conda 时直接复用，不要在每次上课时重新安装。

### 3.1 先检查有没有 Conda

```bash
command -v conda
ls -l "$HOME/miniconda3/bin/conda"
```

第一条能找到 Conda 时继续运行：

```bash
conda info --base
conda env list
```

第二条显示文件不存在，只说明默认位置没有安装；如果第一条已经找到其他合法安装路径，应使用实际路径，不重复安装。

若 `$HOME/miniconda3/bin/conda` 存在但终端找不到 `conda`：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
conda config --set auto_activate_base false
```

关闭此终端并新建一个 Linux 终端，再执行 `conda --version`。`source` 只修复当前终端；`conda init bash` 用于以后新开的 Bash 终端。使用其他安装目录时，相应替换上述路径。

### 3.2 仅尚未安装时执行安装

课程旧服务器固定使用下面的 Linux x86_64 安装包。不要用 Windows 安装包，也不要直接替换成 `latest`。

```bash
mkdir -p "$HOME/installers"
cd "$HOME/installers"
curl -fL -o Miniconda3-py311_24.11.1-0-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Linux-x86_64.sh
echo "807774bae6cd87132094458217ebf713df436f64779faf9bb4c3d4b6615c1e3a  Miniconda3-py311_24.11.1-0-Linux-x86_64.sh" | sha256sum -c -
```

校验输出应为 `OK`。下载失败或校验不一致时先停止，重新获取完整安装包。已有同名安装文件时也先校验，校验通过即可复用，不必重复下载。安装目录已经存在时回到 3.1 核对，不强行覆盖。

校验成功后：

```bash
bash Miniconda3-py311_24.11.1-0-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
conda config --set auto_activate_base false
```

新建一个 Linux 终端，检查初始化是否真正生效：

```bash
conda --version
conda activate base
python -c "import sys; print(sys.executable)"
conda deactivate
```

此时无须再次手动 `source`。Python 路径应属于自己的 Miniconda。`base` 是 Conda 的基础环境，后面为课程单独建立 `dl2026`。

<a id="clone"></a>
## 4. 获取自己的课程代码副本

### 4.1 首次克隆

在 **Linux 终端**，先把公共路径里的 `xxx` 替换为教师公布的值：

```bash
COURSE_REPO="/data/public/xxx/course-repos/Deeplearning2026"
git ls-remote "$COURSE_REPO" refs/heads/main
```

应返回一行提交编号和 `refs/heads/main`。没有结果、找不到仓库或权限不足时，先让教师检查发布路径和权限，不要克隆一个空目录。

然后执行：

```bash
mkdir -p "$HOME/courses"
git clone --no-local -b main "$COURSE_REPO" "$HOME/courses/Deeplearning2026"
```

`--no-local` 避免不同账号之间使用本地硬链接克隆。若提示目标目录已存在，不要删除目录或重复克隆到里面；先进入现有目录检查，它可能就是上次已经下载的项目。

### 4.2 核对副本与更新来源

```bash
cd "$HOME/courses/Deeplearning2026"
pwd
git remote -v
git status --short
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
ls
```

应位于个人课程目录，分支为 `main`，`origin` 指向课堂公共仓库，并能看到 `01_mlp_lab`、`02_cnn_lab`、`requirements.txt` 等文件。新克隆的 `git status --short` 应无输出。

早期直接从 GitHub 克隆的学生，若教师已改用公共仓库分发，先确认当前目录确实是自己的课程副本，再更改来源：

```bash
git remote set-url origin /data/public/xxx/course-repos/Deeplearning2026
git remote -v
```

这一步只用于更换已存在的 `origin`，不需要每次上课执行。教师的工作副本仍保留 GitHub 为 `origin`，不要照做学生这一步。

<a id="environment"></a>
## 5. 创建课程环境、安装依赖并验收

### 5.1 环境只创建一次

先检查：

```bash
conda env list
```

如果已有 `dl2026`，先激活并核对 Python：

```bash
conda activate dl2026
python --version
python -c "import sys; print(sys.executable)"
```

应为 Python 3.11，路径应属于自己的 `dl2026`。环境可用就继续 5.2；版本不符或确实损坏时按[第 8 节](#recovery)确认后重建。

仅没有 `dl2026` 时创建：

```bash
conda create -n dl2026 python=3.11 pip -y
conda activate dl2026
```

`base` 和 `dl2026` 同时存在是正常的，不代表重复安装。`conda env list` 中的 `*` 表示当前环境。

### 5.2 安装课程依赖

在 **Linux 终端**：

```bash
cd "$HOME/courses/Deeplearning2026"
conda activate dl2026
python -m pip --version
python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt
python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

先确认 pip 输出路径属于 `dl2026`。`python -m pip` 可避免调用到其他 Python 的 pip；课程依赖版本统一在 `requirements.txt` 中维护，其中包含 Notebook 所需的 `ipykernel`。

PyTorch 使用现有课堂固定版本 `2.5.1 + cu118`，此包也可以运行 CPU 实验。首次下载包含多个较大的 CUDA 依赖，需要安装目录及临时目录都有足够空间；不要所有人同时反复重装。`--no-cache-dir` 用于本课程磁盘紧张场景，减少下载缓存占用，但不会减少已安装依赖或安装临时空间；以后确需重装时可能重新下载。[pip 缓存说明](https://pip.pypa.io/en/stable/topics/caching/)

安装中断后先看最后的错误。空间不足按第 8 节清理后重跑这两条安装命令即可，已满足版本的包通常会被复用；不必先删除整个环境。不要使用 `sudo pip`、`pip install --user` 或单独把某个课程依赖升级为最新版。

### 5.3 验收安装结果

```bash
python -m pip check
python -c "import sys, torch, ipykernel; print(sys.executable); print('torch:', torch.__version__); print('wheel CUDA:', torch.version.cuda); print(torch.tensor([1, 2]) * 2)"
python check_environment.py
python 01_mlp_lab/test_mlp_smoke.py
```

应看到 `No broken requirements found.`、自己的环境路径、PyTorch `2.5.1`（通常带 `+cu118`）、CUDA 构建版本 `11.8` 及 `tensor([2, 4])`；MLP 检查末尾为 `MLP smoke test passed.`。`torch.version.cuda` 表示包的构建版本，不等于 GPU 当前可用。

上 CNN 课可再运行：

```bash
python 02_cnn_lab/test_cnn_smoke.py
python 02_cnn_lab/test_cnn_composition.py
```

不必每次上课重新安装所有依赖或检查全部实验室。核心实验和课程附带的小型数据可离线运行；首次配置时的下载需要网络。

### 5.4 只有分配了 GPU 才做 GPU 检查

默认保留 `device="cpu"`。如果教师分配了 GPU，先在 Linux 终端查看：

```bash
nvidia-smi
python -c "import torch; print('available:', torch.cuda.is_available()); print('visible count:', torch.cuda.device_count())"
```

在 Notebook 内将实验参数改为教师指定的 `cuda:N`，不要选择 `auto` 或任意空闲卡。若设置了 `CUDA_VISIBLE_DEVICES`，程序看到的是重编号后的逻辑编号：例如只暴露物理卡 3 时，程序通常应选 `cuda:0`。按课堂分配方式核对，不机械地把两个编号混用。

更改终端环境变量不会自动改变一个已经启动的 Notebook 内核。GPU 不可用时先保留 CPU 完成实验，再反馈驱动、分配或内核问题。

<a id="notebook"></a>
## 6. 打开实验文件夹，选内核，保存个人作业

### 6.1 打开完整实验文件夹

在远程 VS Code 选择“文件 → 打开文件夹”。第一次做 MLP 时打开自己的：

```text
$HOME/courses/Deeplearning2026/01_mlp_lab
```

文件选择框若不展开 `$HOME`，先在 Linux 终端执行 `echo "$HOME/courses/Deeplearning2026/01_mlp_lab"`，复制输出的绝对路径。CNN 课对应 `02_cnn_lab`。

建议每次打开当前实验室文件夹，再从左侧资源管理器打开 Notebook。文件夹里应同时有 Notebook 和 `mlp_lab/` 或 `cnn_lab/` 源码包。只打开散落的 `.ipynb`、下载到别处的单个文件，可能找不到项目模块；在终端执行 `cd` 也不会自动改变 VS Code 已打开的文件夹或正在运行的内核目录。

### 6.2 复制个人 Notebook，再修改

官方 Notebook 作为后续更新用的模板。先在同一个实验目录中复制，保留 `我的作业_` 前缀。例如在 **Linux 终端**：

```bash
cd "$HOME/courses/Deeplearning2026/01_mlp_lab"
cp -i "05_自主修改练习.ipynb" "我的作业_学号_姓名_MLP.ipynb"
```

把学号、姓名换成自己的信息。`cp -i` 如果询问是否覆盖已有作业，先回答 `n` 并更换文件名。也可在 VS Code 资源管理器中复制、重命名，然后打开个人副本。

项目 `.gitignore` 已忽略 `我的作业_*.ipynb`、`outputs/`、`logs/` 等个人生成内容。忽略规则用于避免混入课程版本，**不等于备份**。定期把个人 Notebook、报告、结果图、CSV 及修改过的源码下载到自己的电脑。

CNN 作业使用第 05 份数字识别或第 07 份 Fashion 练习，二选一提交；具体要求见 [CNN README](./02_cnn_lab/README.md)。

### 6.3 选择实际运行的内核

打开个人 Notebook，点击右上角“选择内核 / Select Kernel” → “Python Environments” → 选择 `dl2026` 对应的 Python。

注意这三件事分别设置：

- 终端中的 `conda activate dl2026`：决定该终端运行什么 Python。
- `Python: Select Interpreter`：设置 VS Code 的 Python 解释器。
- Notebook 右上角内核：决定这份 `.ipynb` 实际运行什么 Python。

不要因为终端出现 `(dl2026)` 就假定 Notebook 已经选对。此路线不需要自己启动公开的 Jupyter 服务，也不需要填写远程 Jupyter URL 或 token。

在 **Notebook 单元格**执行：

```python
import sys
import socket
from pathlib import Path
import torch

print("服务器：", socket.gethostname())
print("Python：", sys.executable)
print("当前目录：", Path.cwd())
print("PyTorch：", torch.__version__)
print(torch.tensor([1, 2]) * 2)
```

确认主机、环境路径及工作目录正确，再从第一格向下运行。首次体验可以 `Run All`；有“先选择方案再最终测试”的练习，应按 Notebook 中的步骤操作。

若列表没有 `dl2026`，先确认远端 Python/Jupyter 扩展、`conda env list` 和 `python -c "import ipykernel"`。重载窗口后仍无法发现时，可以在已激活的环境里手动注册一次：

```bash
conda activate dl2026
python -m ipykernel install --user --name dl2026 --display-name "Python (dl2026)"
```

这里的 `--user` 是注册个人内核，不是把依赖安装到环境外。随后重新选择内核并用 `sys.executable` 验证。无需每次上课重复注册。

### 6.4 结束实验

保存 Notebook 和需要提交的结果，然后关闭该 Notebook 的内核。部分实验末尾提供“清理并结束内核”单元格，运行后显示内核停止属于正常现象。仅关闭编辑标签或 SSH 窗口不一定立即结束远程 Python 进程，应主动停止不用的内核。

<a id="student-update"></a>
## 7. 学生日常运行与安全更新

### 7.1 每次上课的固定顺序

1. 连接 `course-server`，打开自己的课程或实验室文件夹。
2. 保存上次作业；需要更新时先关闭正在使用旧代码的 Notebook 内核。
3. 在个人仓库根目录检查并拉取课程更新。
4. 根据本次课程进入对应实验室，从新模板复制个人练习。
5. 选择 `dl2026` 内核，从第一格按顺序运行。
6. 保存和备份，结束内核。

第 3 步的 **Linux 终端**命令：

```bash
cd "$HOME/courses/Deeplearning2026"
git remote -v
git rev-parse --abbrev-ref HEAD
git status --short
```

确认 `origin` 是课堂公共仓库、当前在 `main`、没有自己修改过的受跟踪文件。若有修改，先按 7.3 处理，再执行：

```bash
git pull --ff-only origin main
git log -1 --oneline
```

`Already up-to-date` / `Already up to date` 表示学生已经与公共仓库同步，不代表教师的 GitHub 新代码一定已经发布到公共仓库。教师应提供本次发布的提交编号，学生可用 `git rev-parse --short HEAD` 核对。

`--ff-only` 遇到本地历史分叉时会停止，避免不知情地合并。出现 `not possible to fast-forward` 时保留现场并请教师核对，不强制覆盖。

### 7.2 更新之后还要做什么

Git 更新代码文件，Conda 管理 Python 包，二者不是同一个操作。

- 只有教师通知 `requirements.txt` 或 PyTorch 配置变化时，才按[第 5 节](#environment)同步依赖；普通 Notebook / `.py` 修改不需要重装环境。
- 关闭并重新启动内核，再从头运行，让 Python 加载更新后的源码。
- 已复制的“我的作业”Notebook 保留自己的旧内容。它不会自动变成新模板；教师更新练习步骤后，应从新版官方 Notebook 重新复制，按需迁移自己的参数和文字。
- 做实验时记录 `git rev-parse --short HEAD` 的提交编号。旧个人 Notebook 搭配新源码可能出现接口不一致，复现实验还需保留当时使用的代码版本、配置与结果。

### 7.3 不小心修改了官方 Notebook 怎么办

先保存文件并停止内核，在仓库根目录查看：

```bash
git status --short
git diff --name-only
git diff --cached --name-only
```

例如被修改的是 `01_mlp_lab/05_自主修改练习.ipynb`，先复制为个人文件：

```bash
cp -i "01_mlp_lab/05_自主修改练习.ipynb" "01_mlp_lab/我的作业_学号_姓名_MLP_更新前保存.ipynb"
sha256sum "01_mlp_lab/05_自主修改练习.ipynb" "01_mlp_lab/我的作业_学号_姓名_MLP_更新前保存.ipynb"
```

确认没有误覆盖旧作业，两个哈希一致，且能打开保存的副本。然后才将**这一份官方文件**恢复为当前提交的版本：

```bash
git checkout HEAD -- "01_mlp_lab/05_自主修改练习.ipynb"
git status --short
```

这会丢弃指定文件中尚未提交的修改（含已暂存修改），所以必须先备份。其他被修改文件也逐个确认；若改过 `.py`，把源码复制到个人备份目录并保留修改说明，不只备份 Notebook。遇到不清楚用途的状态项，先让教师检查。

全部处理清楚后，再执行 7.1 的 `git pull --ff-only origin main`。不要用整个仓库的 `reset --hard`、`clean -fdx` 或删除项目来解决学生作业冲突。

<a id="recovery"></a>
## 8. 空间不足、重复环境与重装

### 8.1 先判断空间用在哪里

`OSError: [Errno 28] No space left on device` 是存储空间、配额或 inode 等问题，不是模型代码错误。安装日志里的 `Using cached` 表示在复用缓存，不能据此认定每次运行都多存了一整份下载。

在 **Linux 终端**检查：

```bash
conda activate dl2026
python -m pip cache info
conda env list
du -sh "$HOME/.cache/pip" "$HOME/miniconda3/pkgs" "$HOME/miniconda3/envs" 2>/dev/null
df -h "$HOME" "$CONDA_PREFIX"
python -c "import tempfile; print(tempfile.gettempdir())"
```

`du` 里的默认路径不存在时跳过，以 pip 和 Conda 输出的实际路径为准。对最后显示的临时目录再检查磁盘，例如输出 `/tmp` 时：

```bash
df -h /tmp
df -i "$HOME" /tmp
quota -s
```

`quota` 在部分服务器没有安装；没有输出也不能单独证明没有配额限制。如果个人占用不大却安装失败，或公共 `/tmp` / 文件系统已满，把检查结果交给教师或管理员。不要清理其他账号的文件。

### 8.2 先清下载缓存，保留可用环境

确认 `python -m pip cache info` 指向自己账号的缓存，再执行：

```bash
python -m pip cache purge
python -m pip cache info
```

这会移除 pip 下载缓存，不会卸载已经安装的 PyTorch。Conda 安装包缓存仍很大时，可以先预览：

```bash
conda clean --tarballs --dry-run
```

确认列出的都是自己 Conda 安装中的缓存包后，再执行：

```bash
conda clean --tarballs
```

按提示确认即可。本指南只清安装包缓存，不手动删除整个 `pkgs`、`miniconda3` 或共享目录。

清理后回到[5.2 安装依赖](#environment)，使用同样的 `--no-cache-dir` 命令完成安装；之后做 5.3 验收。不必因为一次空间错误立刻删除环境。

### 8.3 删除误建的多余环境

先保存实验并关闭使用该环境的 Notebook 内核和训练进程，再查看：

```bash
conda env list
conda info --base
```

核对环境**名字和完整路径**：`base` 加一个 `dl2026` 是正常配置；同一环境有多个 Jupyter 显示名称也不等于磁盘上有多份 Conda 环境。只删除自己创建、已确认不用的环境。

例如确认 `dl2026_old` 是误建的多余环境：

```bash
conda deactivate
conda env remove -n dl2026_old
conda env list
```

删除前检查提示中的路径，再确认。若要删除的环境仍带 `*`，继续退出该环境再删。多个多余环境逐个核对、逐个删除，不使用批量匹配命令，也不删除 `base`。按路径创建的环境应按 `conda env list` 的实际完整路径核对，可请教师协助使用 `conda env remove -p`。

### 8.4 仅确需重建时删除 dl2026

重建会删除该环境中安装的 Python 包。先确认 Notebook、代码和作业保存在 `$HOME/courses/Deeplearning2026` 等工作目录，而不是环境目录内；有文件放在环境目录里时先备份。

保存作业、停止内核后：

```bash
conda deactivate
conda env list
conda env remove -n dl2026
conda env list
```

确认删除的是自己的 `dl2026`，随后回到[第 5 节](#environment)，依次创建 Python 3.11 环境、安装课程依赖、验收。项目文件仍在原来的个人工作目录，无须重新克隆。

重建后在 VS Code 重新选择内核并检查 `sys.executable`。若之前手动注册过内核，且路径变更后旧入口失效，在新环境激活后重新执行 6.3 的注册命令，再选择新入口。

<a id="teacher"></a>
## 9. 教师如何维护并更新服务器上的项目

本节由教师操作。GitHub 的 `main` 是课程代码发布源，服务器公共仓库是课堂分发入口；学生更新的是公共仓库。**仅向 GitHub 推送，学生的服务器副本不会自动更新。**

教师需要维护两处服务器目录：

| 目录 | 作用 | 谁能写 |
|---|---|---|
| `$HOME/2026DeeplearningDemo` | 教师自己的普通 Git 工作副本，用于同步、检查代码 | 教师账号 |
| `/data/public/xxx/course-repos/Deeplearning2026` | 公共 bare Git 仓库，用于分发提交，不直接编辑或运行 Notebook | 授权发布的教师；学生只读 |

公共仓库目录不带 `.git` 后缀也可以是 bare 仓库。它内部通常是 `HEAD`、`objects`、`refs` 等 Git 数据，没有实验文件工作树。

### 9.1 首次准备教师服务器工作副本

先用教师账号连接服务器，检查：

```bash
whoami
echo "$HOME"
git --version
```

仅在目标目录尚不存在时克隆：

```bash
cd "$HOME"
timeout 180 git clone -b main https://github.com/AYHQChang/2026DeeplearningDemo.git "$HOME/2026DeeplearningDemo"
```

已有目录时直接进入并检查，不重新覆盖：

```bash
cd "$HOME/2026DeeplearningDemo"
git rev-parse --show-toplevel
git remote -v
git rev-parse --abbrev-ref HEAD
git status --short
```

确认是教师专用副本、`main` 分支、`origin` 指向上述 GitHub 仓库。若这里有尚未保存的教学修改，先备份或提交到正常维护流程，不直接抹掉。

### 9.2 首次建立公共分发仓库

实际公共路径由服务器管理员分配，教师账号必须有写权限。先核对路径和父目录：

```bash
COURSE_REPO="/data/public/xxx/course-repos/Deeplearning2026"
id
ls -ld /data /data/public /data/public/xxx
```

父目录不存在或没有权限时，先由管理员准备授权目录。不要为了发布课程代码把教师整个主目录改成公共可写，也不要使用 `chmod 777`。

确认父目录授权正确、目标目录尚不存在时：

```bash
mkdir -p /data/public/xxx/course-repos
git init --bare --shared=0644 "$COURSE_REPO" && git --git-dir="$COURSE_REPO" symbolic-ref HEAD refs/heads/main
```

`--shared=0644` 适合单一教师账号发布、其他账号读取的课程仓库。父目录仍需允许学生遍历；既有 ACL 或服务器权限策略也可能影响访问，应做独立账号验收。[Git 初始化与共享权限说明](https://git-scm.com/docs/git-init)

目标已经存在时只检查，不重复初始化：

```bash
git --git-dir="$COURSE_REPO" rev-parse --is-bare-repository
git --git-dir="$COURSE_REPO" symbolic-ref HEAD
```

预期为 `true` 和 `refs/heads/main`。如果不是 bare 仓库，先确认目录用途，不把普通代码目录直接转换为公共仓库。首次空仓库还没有 `main` 提交，需完成后面的发布才能供学生克隆。

### 9.3 每次更新：先把本地修改发布到 GitHub

在教师平时维护代码的电脑上完成修改和对应实验检查。用 Git 工具检查差异，明确选择本次要发布的代码、Notebook、文档和资源文件；新建但未加入 Git 的文件不会随推送发布。

例如在教师本机 **Windows PowerShell**（实际本地目录以自己的为准）：

```powershell
Set-Location 'D:\Code_for_all\2026DeeplearningDemo'
git status --short
git diff --stat
git diff --check
```

检查后选择相关文件暂存，再提交、推送。下面示例只表示本次维护这份指南；有配套修改时应一并选择相应文件，而不是照抄成只提交指南：

```powershell
git add -- '从零到运行Notebook-服务器实验完整操作指南.md'
git diff --cached --stat
git diff --cached
git commit -m "docs: update server operation guide"
git push origin main
git rev-parse HEAD
```

推送失败时先处理，不能把本地提交号当成已发布版本。准备同步到服务器前，记录 GitHub `main` 已成功接收的完整提交编号。不要提交学生个人作业、密码、服务器私有配置或运行缓存。

### 9.4 每次更新：同步教师服务器工作副本

在教师账号的 **Linux 终端**：

```bash
cd "$HOME/2026DeeplearningDemo"
git rev-parse --show-toplevel
git remote -v
git rev-parse --abbrev-ref HEAD
git status --short
```

必须确认位置正确、`origin` 是 GitHub、分支为 `main`，并处理完所有需要保留的工作区修改。正常更新使用：

```bash
timeout 90 git fetch origin && git merge --ff-only origin/main
```

只有获取成功才会继续合并。超时、认证失败或无法快进时停止发布，先解决原因；不要把旧的 `origin/main` 缓存误当成本次远端版本。

检查同步结果：

```bash
git rev-parse HEAD
git rev-parse origin/main
git log -1 --oneline
git status --short
```

两个完整提交号应相同，并与 9.3 记录的 GitHub 发布编号一致；有更新竞争时重新核对目标版本。

如果教师过去在服务器副本里临时修改或提交过代码，导致不能快进，先保留需要的内容。**仅对已确认可覆盖的教师专用同步副本**，可以使用原先的强制对齐方式：

```bash
timeout 90 git fetch origin && git reset --hard origin/main
```

这会丢弃该副本中受跟踪文件的未提交修改，并将当前分支对齐远端；不要在学生作业目录使用。它不会清理所有未跟踪或被忽略的文件，运行检查前仍应核对是否存在旧实验产物影响结果。

### 9.5 发布前检查：代码与环境分别处理

教师同样使用自己的 `dl2026` 环境。教师工作副本路径与学生不同，从下面目录运行：

```bash
cd "$HOME/2026DeeplearningDemo"
conda activate dl2026
python check_environment.py
```

首次配置教师环境也按第 3、5 节进行，但把工作目录换成教师副本。若本次改变 `requirements.txt`，先在这里安装更新依赖、通过检查，再通知学生同步；安装到教师环境不会更新学生环境。

根据本次修改选择相应检查，不必文档改动也运行所有训练。例如 CNN 功能发布：

```bash
python 02_cnn_lab/test_cnn_smoke.py
python 02_cnn_lab/test_cnn_composition.py
```

各实验室已有检查入口如下，均从仓库根目录执行：

| 修改范围 | 检查入口 |
|---|---|
| MLP 分类 | `python 01_mlp_lab/test_mlp_smoke.py` |
| 房价回归 | `python 01_mlp_lab/test_house_price_smoke.py` |
| CNN | 上述两条 CNN 检查 |
| RNN | `python 03_rnn_lab/test_rnn_smoke.py` |
| Transformer | `python 04_transformer_lab/test_transformer_smoke.py` |
| 统一速度入口 | `python test_device_speed_demo.py` |

Notebook 的教学流程发生变化时，还要在远程 VS Code 打开对应模板，选对内核，从头试运行相关流程。新练习只验证函数还不够，要确认学生能找到可编辑位置、形成结果并保存作业。

### 9.6 发布到公共仓库，并核对提交号

检查通过后，在教师副本中执行（每个新终端重新设置实际公共路径）：

```bash
cd "$HOME/2026DeeplearningDemo"
COURSE_REPO="/data/public/xxx/course-repos/Deeplearning2026"
git push "$COURSE_REPO" main:refs/heads/main
```

普通推送足够。若提示 `non-fast-forward`，先检查公共仓库是否已有其他教师的新提交或发布来源是否正确，不直接加 `--force`。

发布成功后核对：

```bash
git rev-parse main
git --git-dir="$COURSE_REPO" rev-parse refs/heads/main
git --git-dir="$COURSE_REPO" log -1 --oneline
```

教师副本、公共仓库的提交号应等于本次 GitHub 发布编号。首次发布再确认 `git --git-dir="$COURSE_REPO" symbolic-ref HEAD` 为 `refs/heads/main`。

### 9.7 用独立学生账号验收，并说明更新事项

教师账号能读写不代表学生有权限。首次部署或调整权限后，请一名学生用自己的账号执行：

```bash
git ls-remote /data/public/xxx/course-repos/Deeplearning2026 refs/heads/main
```

学生按第 4 节首次克隆，已有副本按第 7 节更新。核对学生端 `git rev-parse HEAD` 等于发布编号，并至少打开本节课 Notebook 运行入口；如无法读取，检查公共路径各级目录的遍历权限和仓库读取权限，由教师或管理员修正，不让学生改公共仓库权限。

每次发布向学生说明四件事：本次提交编号、修改了哪份 Notebook、是否需要同步依赖、是否要从新模板重新复制个人作业。提醒更新前保存作业，更新后重启内核。

### 9.8 服务器访问 GitHub 超时怎么办

`timeout` 到时退出码通常为 `124`；需要查看时在失败命令后立即执行 `echo $?`。先核对网络，不连续执行推送来掩盖同步失败。

若教师电脑能访问 GitHub，也能 SSH 到课程服务器，可以从本机把已经发布到 GitHub 的同一提交直接推送到公共仓库。先在本机课程目录执行：

```powershell
git fetch origin
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git rev-parse origin/main
```

确认获取成功、处于干净的 `main`，两个提交号一致。为教师账号配置 SSH 别名，例如 `course-teacher`（与 2.2 相同格式，但使用教师账号和实际端口），再执行：

```powershell
git push 'course-teacher:/data/public/xxx/course-repos/Deeplearning2026' main:refs/heads/main
```

需要教师账号对既有公共 bare 仓库有写权限。仍不使用强制推送，完成后按 9.6–9.7 核对公共仓库和学生副本。此方法不会更新教师服务器工作副本；网络恢复后再按 9.4 同步它，避免以后从旧副本发布。

<a id="troubleshooting"></a>
## 10. 常见问题按报错定位

| 现象 | 先检查什么 | 对应处理 |
|---|---|---|
| SSH 超时、拒绝连接 | 网络、地址、端口；是否已到登录阶段 | [2.4](#connect) |
| SSH 成功，VS Code 报 GLIBC / LinuxPrereqs | 客户端版本与旧服务器系统 | 使用课堂兼容客户端，见第 2 节 |
| `conda: command not found` | 是否装过、实际安装位置、初始化 | [第 3 节](#conda)，不要立即重装 |
| `CondaError: Run conda init before conda activate` | 当前 Bash 是否初始化 | `source` 后 `conda init bash`，新开终端验证 |
| `No module named torch` / `ipykernel` | Notebook 的 `sys.executable` 和安装包的环境是否一致 | [第 5、6 节](#environment) |
| `No module named mlp_lab` / `cnn_lab` / `core` | 是否打开完整实验文件夹，源码包是否存在 | 重新打开对应 lab 文件夹、重启内核，从首格运行 |
| Notebook 只有选择远程 Jupyter URL 的入口 | 远端 Python/Jupyter 扩展、解释器发现状态 | 选择 Python Environments，见第 6 节 |
| `No space left on device` | pip 缓存、重复环境、配额、临时目录、inode | [第 8 节](#recovery) |
| `Your local changes ... would be overwritten` | 是否修改了官方文件 | 先备份，再逐个恢复，见 7.3 |
| `Already up to date` 却没有新内容 | 教师是否发布公共仓库；自己是否看的是旧个人 Notebook | 比较提交号，再从新模板复制 |
| 更新后仍是旧函数或旧图 | 内核是否仍加载旧代码 | 重启内核，从头运行；已有图片不会自动重画 |
| GPU 不可用 / invalid device ordinal | 内核环境、可见设备数、物理与逻辑编号 | 按分配核对，先用 CPU；见 5.4 |
| 结束实验后显示内核已停止 | 是否运行了“清理并结束内核”单元格 | 正常，需要继续时重新启动 |

`mlp_lab`、`cnn_lab` 是本项目附带的源码包，不是需要从 pip 下载的同名第三方包。更新 Git 只改变磁盘文件，不会自动修正已启动内核的工作目录；这就是“重新打开 01_mlp_lab 文件夹后能导入”的原因。

反馈问题时附上：完整报错最后一段、发生在哪一步、`pwd`、`git log -1 --oneline`、`conda env list`，以及 Notebook 中 `sys.executable` 与 `Path.cwd()` 的输出。不要只截最后一个 `Error`，也不要发送密码。

<a id="reference"></a>
## 11. 命令速查与完成标准

以下是 Linux 终端常用命令；`python -c` 可在终端执行，Notebook 中直接写 Python 代码。

| 目的 | 命令 |
|---|---|
| 我是谁、在哪台机器 | `whoami`、`hostname` |
| 当前目录 / 主目录 | `pwd`、`echo "$HOME"` |
| 查看目录里的文件 | `ls`、`ls -la` |
| 回到学生课程根目录 | `cd "$HOME/courses/Deeplearning2026"` |
| 进入当前目录下的 CNN 实验室 | `cd 02_cnn_lab` |
| 返回上一层目录 | `cd ..` |
| 列出环境 / 进入 / 退出 | `conda env list`、`conda activate dl2026`、`conda deactivate` |
| 当前 Python 路径 | `python -c "import sys; print(sys.executable)"` |
| 查看安装的包 / 检查依赖 | `python -m pip list`、`python -m pip check` |
| 查当前版本 / 未提交修改 | `git log -1 --oneline`、`git status --short` |
| 查课程更新来源 | `git remote -v` |
| 终端停止当前程序 | `Ctrl+C`；Notebook 使用停止按钮并按需关闭内核 |

第一次配置完成后，应能独立做到：连接自己的服务器账号；新开终端激活 `dl2026`；从个人课程副本打开当前实验室；在正确内核里运行个人 Notebook 并保存结果。

教师一次发布完成的标准是：GitHub、教师服务器副本、公共仓库具有目标提交，学生能从公共仓库获取该提交并运行本次课程入口。使用 9.8 的临时直推方式时，单独记录教师服务器副本待同步。

后续服务器连接、环境、更新和清理流程统一维护本文件。各实验室 README 保留实验使用说明；被替代的独立笔记原样保存在[2026-10-07 归档](./docs/archive/2026-10-07/README.md)，仅用于历史查阅。
