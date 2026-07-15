"""智能 Nuitka 打包脚本 —— 根据 CPU / 内存自动配置最优并行编译参数。"""
import os
import sys
import subprocess
from pathlib import Path

# 切换到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def detect_hardware():
    """检测硬件并返回 (cpu_logical, ram_total_gb, ram_avail_gb)."""
    info = {"cpu_logical": os.cpu_count() or 4, "ram_total_gb": 0, "ram_avail_gb": 0}

    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = vm.total / (1024 ** 3)
        info["ram_avail_gb"] = vm.available / (1024 ** 3)
        info["cpu_logical"] = psutil.cpu_count(logical=True) or info["cpu_logical"]
    except ImportError:
        info["ram_total_gb"] = 8
        info["ram_avail_gb"] = 4

    return info


def compute_jobs(cpu_logical, ram_avail_gb):
    """每个 C 编译作业保守估算 500MB 内存，留 2GB 给链接器和系统。"""
    ram_based = max(1, int((ram_avail_gb - 2) / 0.5))
    return min(cpu_logical, ram_based)


def build_nuitka_args(info):
    jobs = compute_jobs(info["cpu_logical"], info["ram_avail_gb"])

    args = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        f"--jobs={jobs}",
        "--windows-console-mode=disable",
        "--show-progress",
        "--module-parameter=torch-disable-jit=yes",
        "--enable-plugin=pyside6",
        "--include-qt-plugins=multimedia",
        "--include-package=ultralytics",
        "--include-package=processor",
        "--include-package=gui",
        "--include-package=logging_config",
        "--include-package=torch.cuda",
        "--assume-yes-for-downloads",
    ]

    if info["ram_total_gb"] < 8:
        args.append("--lto=no")

    data_files = [
        ("weights/best.pt", "./weights/best.pt"),
        ("weights/best_imgsz_1280.onnx", "./weights/best_imgsz_1280.onnx"),
        ("config.yml", "./config.yml"),
        ("openh264-1.8.0-win64.dll", "./openh264-1.8.0-win64.dll"),
    ]
    for src, dst in data_files:
        if os.path.exists(src):
            args.append(f"--include-data-files={src}={dst}")

    args.extend(["--output-dir=dist", "main.py"])
    return args, jobs


def main():
    info = detect_hardware()
    args, jobs = build_nuitka_args(info)

    print("=" * 56)
    print("  智能 Nuitka 打包 —— 硬件检测 & 参数优化")
    print("=" * 56)
    print(f"  CPU 逻辑核心 : {info['cpu_logical']}")
    print(f"  总内存       : {info['ram_total_gb']:.1f} GB")
    print(f"  可用内存     : {info['ram_avail_gb']:.1f} GB")
    print(f"  并行编译作业 : {jobs}")
    print(f"  LTO          : {'否' if info['ram_total_gb'] < 8 else '是 (auto)'}")
    print(f"  ccache       : 是 (自动下载)")
    print(f"  torch.cuda   : 始终包含 (打包产物自适应 CPU/GPU)")
    print("=" * 56)
    print()

    answer = input("开始打包？[Y/n] ").strip().lower()
    if answer and answer != "y" and answer != "yes":
        print("已取消。")
        return

    print("\n[info] 启动 Nuitka 编译...")
    print(f"[cmd] python -m nuitka --standalone --jobs={jobs} ...\n")
    sys.stdout.flush()

    rc = subprocess.run(args, check=False).returncode

    print()
    if rc == 0:
        print("=" * 56)
        print("  打包成功！")
        print(f"  产物目录: dist{os.sep}main.dist{os.sep}")
        print(f"  可执行文件: dist{os.sep}main.dist{os.sep}main.exe")
        print("=" * 56)
    else:
        print("=" * 56)
        print(f"  打包失败 (exit code: {rc})")
        print("=" * 56)

    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
