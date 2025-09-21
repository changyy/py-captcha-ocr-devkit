#!/usr/bin/env python3
"""
CAPTCHA OCR 開發助手 - 主入口文件

使用方式:
  python main.py --help                                     # 查看幫助
  python main.py train --input ./data --output ./models    # 訓練模型
  python main.py evaluate --input ./test --model ./models/best_model.pth  # 評估模型
  python main.py api --model ./models/best_model.pth       # 啟動 API 服務
  python main.py generate --text "abcd" --output ./test.png # 生成圖片
"""

import sys
import os

# 將 src 目錄添加到 Python 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 導入 CLI 主模組
from captcha_ocr_devkit.cli.main import cli

if __name__ == '__main__':
    cli()