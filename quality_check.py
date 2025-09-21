#!/usr/bin/env python3
"""
CAPTCHA OCR Develop Helper - 專案品質驗證腳本

用途：為新用戶提供一鍵式專案健康度檢查
執行：python quality_check.py
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 顏色輸出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """印製標題"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(text: str):
    """印製成功訊息"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """印製錯誤訊息"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    """印製警告訊息"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    """印製資訊訊息"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

def run_command(cmd: List[str], cwd: Path = None) -> Tuple[bool, str, str]:
    """執行命令並返回結果"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "命令執行超時"
    except Exception as e:
        return False, "", str(e)

def check_environment():
    """檢查環境設置"""
    print_header("環境檢查")

    # 檢查 Python 版本
    python_version = sys.version_info
    if python_version >= (3, 8):
        print_success(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print_error(f"Python 版本過低: {python_version.major}.{python_version.minor}.{python_version.micro} (需要 >= 3.8)")
        return False

    # 檢查虛擬環境
    venv_path = Path("./venv")
    if venv_path.exists():
        print_success("虛擬環境已建立")
    else:
        print_error("虛擬環境不存在，請執行: python -m venv venv")
        return False

    # 檢查安裝
    cli_path = venv_path / "bin" / "captcha-ocr-devkit"
    if not cli_path.exists():
        cli_path = venv_path / "Scripts" / "captcha-ocr-devkit.exe"  # Windows

    if cli_path.exists():
        print_success("CLI 工具已安裝")
    else:
        print_error("CLI 工具未安裝，請執行: ./venv/bin/pip install -e .")
        return False

    return True

def check_handler_discovery():
    """檢查 Handler 發現機制"""
    print_header("Handler 發現機制檢查")

    venv_python = Path("./venv/bin/python3")
    if not venv_python.exists():
        venv_python = Path("./venv/Scripts/python.exe")  # Windows

    cmd = [
        str(venv_python), "-c",
        """
from captcha_ocr_devkit.core.handlers.registry import registry
from pathlib import Path
discovered = registry.discover_handlers(Path('./handlers'))
print('發現的 handlers:', discovered)
for handler_type, handlers in discovered.items():
    print(f'{handler_type}: {len(handlers)} 個')
"""
    ]

    success, stdout, stderr = run_command(cmd)
    if success:
        print_success("Handler 發現機制正常")
        print_info(f"輸出：\n{stdout.strip()}")
        return True
    else:
        print_error(f"Handler 發現失敗: {stderr}")
        return False

def check_init_functionality():
    """檢查 init 功能"""
    print_header("Init 功能檢查")

    test_dir = Path("./quality_test_handlers")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    cli_path = Path("./venv/bin/captcha-ocr-devkit")
    if not cli_path.exists():
        cli_path = Path("./venv/Scripts/captcha-ocr-devkit.exe")

    cmd = [str(cli_path), "init", "--output-dir", str(test_dir), "--force"]
    success, stdout, stderr = run_command(cmd)

    if success and test_dir.exists():
        handler_files = list(test_dir.glob("*.py"))
        print_success(f"Init 功能正常，生成了 {len(handler_files)} 個 handler 檔案")

        # 清理
        shutil.rmtree(test_dir)
        return True
    else:
        print_error(f"Init 功能失敗: {stderr}")
        return False

def check_train_functionality():
    """檢查訓練功能"""
    print_header("訓練功能檢查")

    # 創建測試資料
    test_images_dir = Path("./quality_test_images")
    if test_images_dir.exists():
        shutil.rmtree(test_images_dir)

    test_images_dir.mkdir()

    # 創建假圖片檔案
    test_files = [
        "abcd_001.png", "abcd_002.png", "abcd_003.png",
        "efgh_001.png", "efgh_002.png",
        "ijkl_001.png"
    ]

    for filename in test_files:
        (test_images_dir / filename).write_text(f"fake image data for {filename}")

    print_info(f"創建了 {len(test_files)} 個測試圖片檔案")

    # 執行訓練
    cli_path = Path("./venv/bin/captcha-ocr-devkit")
    if not cli_path.exists():
        cli_path = Path("./venv/Scripts/captcha-ocr-devkit.exe")

    model_output = Path("./quality_test_model.json")
    cmd = [
        str(cli_path), "train",
        "--input", str(test_images_dir),
        "--output", str(model_output),
        "--handler", "demo_train",
        "--epochs", "1",
        "--validation-split", "0.2"
    ]

    success, stdout, stderr = run_command(cmd)

    if success and model_output.exists():
        # 檢查模型檔案內容
        try:
            with open(model_output, 'r') as f:
                model_data = json.load(f)

            required_keys = ["model_type", "training_config", "dataset_info"]
            missing_keys = [key for key in required_keys if key not in model_data]

            if not missing_keys:
                print_success("訓練功能正常，模型檔案格式正確")
                print_info(f"處理了 {model_data.get('dataset_info', {}).get('total_images', 'unknown')} 張圖片")
                print_info(f"發現 {model_data.get('dataset_info', {}).get('unique_labels', 'unknown')} 個標籤")
                result = True
            else:
                print_error(f"模型檔案缺少必要欄位: {missing_keys}")
                result = False

        except json.JSONDecodeError:
            print_error("模型檔案格式無效")
            result = False
    else:
        print_error(f"訓練失敗: {stderr}")
        result = False

    # 清理
    if test_images_dir.exists():
        shutil.rmtree(test_images_dir)
    if model_output.exists():
        model_output.unlink()

    return result

def check_evaluate_functionality():
    """檢查評估功能"""
    print_header("評估功能檢查")

    # 使用現有的測試模型
    test_model = Path("./test_model.json")
    test_images_dir = Path("./test_images")

    if not test_model.exists():
        print_warning("測試模型不存在，跳過評估檢查")
        return True

    if not test_images_dir.exists():
        print_warning("測試圖片目錄不存在，跳過評估檢查")
        return True

    cli_path = Path("./venv/bin/captcha-ocr-devkit")
    if not cli_path.exists():
        cli_path = Path("./venv/Scripts/captcha-ocr-devkit.exe")

    cmd = [
        str(cli_path), "evaluate",
        "--target", str(test_images_dir),
        "--model", str(test_model),
        "--handler", "demo_evaluate"
    ]

    success, stdout, stderr = run_command(cmd)

    if success:
        print_success("評估功能正常")
        if "準確率" in stdout:
            print_info("評估結果包含準確率統計")
        return True
    else:
        print_error(f"評估失敗: {stderr}")
        return False

def check_api_functionality():
    """檢查 API 功能"""
    print_header("API 功能檢查")

    test_model = Path("./test_model.json")
    if not test_model.exists():
        print_warning("測試模型不存在，跳過 API 檢查")
        return True

    cli_path = Path("./venv/bin/captcha-ocr-devkit")
    if not cli_path.exists():
        cli_path = Path("./venv/Scripts/captcha-ocr-devkit.exe")

    # 啟動 API 服務
    cmd = [
        str(cli_path), "api",
        "--model", str(test_model),
        "--port", "54399",
        "--handler", "demo_ocr",
        "--preprocess-handler", "demo_preprocess"
    ]

    print_info("啟動 API 服務...")

    try:
        # 在背景啟動 API
        api_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 等待服務啟動
        time.sleep(5)

        # 檢查健康狀態
        try:
            response = requests.get("http://localhost:54399/api/v1/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print_success("API 健康檢查通過")
                print_info(f"服務狀態: {health_data.get('status', 'unknown')}")

                # 測試 OCR 端點
                test_image_data = b"fake image data"
                files = {"file": ("test.png", test_image_data, "image/png")}

                try:
                    ocr_response = requests.post(
                        "http://localhost:54399/api/v1/ocr",
                        files=files,
                        timeout=10
                    )

                    if ocr_response.status_code == 200:
                        ocr_data = ocr_response.json()
                        if "status" in ocr_data and "data" in ocr_data:
                            print_success("OCR API 端點正常")
                            print_info(f"回應格式包含必要欄位: status, data")
                            result = True
                        else:
                            print_error("OCR API 回應格式不正確")
                            result = False
                    else:
                        print_error(f"OCR API 呼叫失敗: {ocr_response.status_code}")
                        result = False

                except requests.RequestException as e:
                    print_error(f"OCR API 測試失敗: {e}")
                    result = False

            else:
                print_error(f"API 健康檢查失敗: {response.status_code}")
                result = False

        except requests.RequestException as e:
            print_error(f"無法連接到 API 服務: {e}")
            result = False

    finally:
        # 關閉 API 服務
        try:
            api_process.terminate()
            api_process.wait(timeout=5)
            print_info("API 服務已關閉")
        except:
            api_process.kill()

    return result

def generate_quality_report(results: Dict[str, bool]):
    """生成品質報告"""
    print_header("品質檢查報告")

    total_checks = len(results)
    passed_checks = sum(results.values())

    print(f"\n{Colors.BOLD}總檢查項目: {total_checks}{Colors.END}")
    print(f"{Colors.BOLD}通過項目: {Colors.GREEN}{passed_checks}{Colors.END}")
    print(f"{Colors.BOLD}失敗項目: {Colors.RED}{total_checks - passed_checks}{Colors.END}")

    print(f"\n{Colors.BOLD}詳細結果:{Colors.END}")
    for check_name, passed in results.items():
        status = f"{Colors.GREEN}✅ 通過{Colors.END}" if passed else f"{Colors.RED}❌ 失敗{Colors.END}"
        print(f"  {check_name}: {status}")

    success_rate = (passed_checks / total_checks) * 100

    if success_rate == 100:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 專案品質檢查全部通過！{Colors.END}")
        print(f"{Colors.GREEN}專案處於健康狀態，可以開始使用。{Colors.END}")
    elif success_rate >= 80:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  專案品質良好，但有部分問題需要修復。{Colors.END}")
        print(f"{Colors.YELLOW}成功率: {success_rate:.1f}%{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ 專案存在較多問題，需要修復後再使用。{Colors.END}")
        print(f"{Colors.RED}成功率: {success_rate:.1f}%{Colors.END}")

    return success_rate >= 80

def main():
    """主函數"""
    print_header("CAPTCHA OCR Develop Helper - 品質檢查")
    print_info("開始進行專案品質檢查...")

    # 確保在正確的目錄中
    if not Path("setup.py").exists():
        print_error("請在專案根目錄執行此腳本")
        sys.exit(1)

    results = {}

    # 執行各項檢查
    results["環境檢查"] = check_environment()
    results["Handler 發現機制"] = check_handler_discovery()
    results["Init 功能"] = check_init_functionality()
    results["訓練功能"] = check_train_functionality()
    results["評估功能"] = check_evaluate_functionality()
    results["API 功能"] = check_api_functionality()

    # 生成報告
    overall_success = generate_quality_report(results)

    # 提供後續建議
    print_header("後續建議")

    if overall_success:
        print_info("專案已準備就緒，你可以：")
        print("  1. 執行 './venv/bin/captcha-ocr-devkit init' 創建你的 handlers")
        print("  2. 準備你的訓練資料並執行訓練")
        print("  3. 使用 API 服務進行 CAPTCHA 識別")
        print("  4. 查看 CLAUDE.md 了解更多用法")
    else:
        print_warning("請先修復失敗的檢查項目：")
        for check_name, passed in results.items():
            if not passed:
                print(f"  - {check_name}")
        print("\n修復後重新執行此腳本進行驗證。")

    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())
