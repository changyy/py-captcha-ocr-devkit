"""Setup script for captcha-ocr-devkit."""

from setuptools import setup, find_packages
from pathlib import Path

# 讀取 README
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

def load_core_requirements() -> list[str]:
    requirements_file = this_directory / "requirements.txt"
    if not requirements_file.exists():
        return []

    requirements: list[str] = []
    for line in requirements_file.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        requirements.append(cleaned)
    return requirements


# 核心輕量級依賴
core_requirements = load_core_requirements()

def load_version() -> str:
    version_file = this_directory / "src" / "captcha_ocr_devkit" / "__init__.py"
    namespace: dict[str, str] = {}
    exec(version_file.read_text(encoding="utf-8"), namespace)
    return namespace.get("__version__", "0.0.0")


extras_require = {
    # 基本圖片處理
    "pillow": [
        "Pillow>=9.0.0",
    ],

    # OpenCV 圖片處理
    "opencv": [
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
    ],

    # PyTorch 機器學習
    "pytorch": [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
    ],

    # TensorFlow 機器學習
    "tensorflow": [
        "tensorflow>=2.10.0",
        "numpy>=1.21.0",
    ],

    # Tesseract OCR
    "tesseract": [
        "pytesseract>=0.3.10",
        "Pillow>=9.0.0",
    ],

    # 資料分析和視覺化
    "viz": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "numpy>=1.21.0",
    ],

    # 開發和測試
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "isort>=5.12.0",
        "mypy>=1.0.0",
        "tqdm>=4.64.0",
    ],

    # 文檔
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
}


_runtime_bundles = ["pillow", "opencv", "pytorch", "tensorflow", "tesseract", "viz"]
extras_require["all"] = sorted({dep for key in _runtime_bundles for dep in extras_require[key]})

setup(
    name="captcha-ocr-devkit",
    version=load_version(),
    author="changyy",
    author_email="changyy.csie@gmail.com",
    description="插件化 CAPTCHA OCR 開發套件 - 輕量級、可擴展、自定義 handlers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/changyy/py-captcha-ocr-devkit",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "captcha_ocr_devkit.api": ["demo_index.html"],
        "captcha_ocr_devkit.examples.handlers": [
            "*-requirements.txt",
            "*-requirements-*.txt",
            "*-README.md",
            "*-config.json",
        ],
        "captcha_ocr_devkit.examples.scripts": ["*.sh"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Framework :: FastAPI",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "captcha-ocr-devkit=captcha_ocr_devkit.cli.main:cli",
            "captcha-ocr-helper=captcha_ocr_devkit.cli.main:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "captcha",
        "ocr",
        "plugin-architecture",
        "handlers",
        "extensible",
        "lightweight",
        "computer-vision",
        "fastapi",
        "cli-tool",
        "text-recognition",
        "image-processing"
    ],
    project_urls={
        "Bug Reports": "https://github.com/changyy/py-captcha-ocr-devkit/issues",
        "Source": "https://github.com/changyy/py-captcha-ocr-devkit",
        "Documentation": "https://github.com/changyy/py-captcha-ocr-devkit#readme",
    },
)
