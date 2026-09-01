"""深度学习课堂 Demo 的本地环境自检脚本。

脚本不联网、不安装或升级任何包，只完成以下检查：

1. 打印当前 Python 版本，确认 VS Code/Jupyter 选中了预期 Conda 环境；
2. 逐个导入训练、绘图与 Notebook 执行真正需要的 Python 模块；
3. 确认 Jupyter 与 nbconvert 已安装，并输出每个发行包的实际版本；
4. 检查 CUDA 是否可用，并在可用时打印 CUDA runtime 与 GPU 名称。

只要有一个必需包缺失，脚本就以非零状态退出并列出缺失名称；全部通过后才输出
“环境自检通过”。该文件不检查网络，因为正式课堂运行设计为完全离线。
"""

from __future__ import annotations

from importlib import metadata, util
import sys


IMPORT_CHECKS = {
    "torch": "torch",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "nbformat": "nbformat",
    "ipykernel": "ipykernel",
    "nbclient": "nbclient",
}

# 这两个发行包只需确认已安装。环境自检不启动 Jupyter 服务，也不联网。
INSTALL_CHECKS = {
    "jupyter": "jupyter_core",
    "nbconvert": "nbconvert",
}


def main() -> None:
    """执行依赖导入、版本查询与 CUDA 状态检查。

    Returns:
        ``None``。

    Raises:
        SystemExit: 一个或多个依赖缺失时，以包含缺失包名称的中文消息退出。
    """

    print(f"Python：{sys.version.split()[0]}")
    missing: list[str] = []
    for package, import_name in IMPORT_CHECKS.items():
        try:
            __import__(import_name)
            print(f"[OK] {package} {metadata.version(package)}")
        except (ImportError, metadata.PackageNotFoundError):
            missing.append(package)
            print(f"[缺失] {package}")

    for package, import_name in INSTALL_CHECKS.items():
        try:
            if util.find_spec(import_name) is None:
                raise ModuleNotFoundError(import_name)
            print(f"[OK] {package} {metadata.version(package)}")
        except (ImportError, metadata.PackageNotFoundError, ModuleNotFoundError):
            missing.append(package)
            print(f"[缺失] {package}")

    if missing:
        raise SystemExit("环境尚未就绪，缺失：" + ", ".join(missing))

    import torch

    print(f"CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime：{torch.version.cuda}")
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print("环境自检通过，可以离线运行课堂 Demo。")


if __name__ == "__main__":
    main()
