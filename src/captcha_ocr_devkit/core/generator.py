"""
CAPTCHA 圖片生成器
負責生成各種樣式的 CAPTCHA 驗證碼圖片
"""

import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from typing import Tuple, List, Optional, Dict, Union
import io
import base64
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class CaptchaGenerator:
    """
    CAPTCHA 圖片生成器
    支援多種樣式和干擾效果
    """

    def __init__(self,
                 width: int = 128,
                 height: int = 64,
                 font_size: int = 24,
                 character_count: int = 4):
        self.width = width
        self.height = height
        self.font_size = font_size
        self.character_count = character_count

        # 字元集合
        self.alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
        self.colors = [
            (0, 0, 0),      # 黑色
            (255, 0, 0),    # 紅色
            (0, 128, 0),    # 綠色
            (0, 0, 255),    # 藍色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
            (128, 128, 128) # 灰色
        ]

        # 背景顏色
        self.background_colors = [
            (255, 255, 255),  # 白色
            (240, 240, 240),  # 淺灰
            (255, 255, 224),  # 淺黃
            (240, 248, 255),  # 淺藍
        ]

    def generate_text(self, length: Optional[int] = None) -> str:
        """
        生成隨機文字

        Args:
            length: 文字長度，預設使用 character_count

        Returns:
            隨機生成的文字
        """
        if length is None:
            length = self.character_count

        return ''.join(random.choices(self.alphabet, k=length))

    def _get_random_font(self, size: Optional[int] = None) -> ImageFont.ImageFont:
        """
        獲取隨機字體

        Args:
            size: 字體大小

        Returns:
            字體物件
        """
        if size is None:
            size = self.font_size

        # 嘗試載入系統字體
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception:
                    continue

        # 使用預設字體
        try:
            return ImageFont.load_default()
        except Exception:
            return ImageFont.ImageFont()

    def _add_noise_lines(self, draw: ImageDraw.Draw, image_size: Tuple[int, int], count: int = 3):
        """
        添加干擾線條

        Args:
            draw: 繪圖物件
            image_size: 圖片尺寸
            count: 線條數量
        """
        width, height = image_size

        for _ in range(count):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            color = random.choice(self.colors)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 2))

    def _add_noise_dots(self, draw: ImageDraw.Draw, image_size: Tuple[int, int], count: int = 50):
        """
        添加干擾點

        Args:
            draw: 繪圖物件
            image_size: 圖片尺寸
            count: 點的數量
        """
        width, height = image_size

        for _ in range(count):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            color = random.choice(self.colors)
            draw.point((x, y), fill=color)

    def _distort_image(self, image: Image.Image) -> Image.Image:
        """
        扭曲圖片

        Args:
            image: 原始圖片

        Returns:
            扭曲後的圖片
        """
        # 隨機旋轉
        angle = random.uniform(-15, 15)
        image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))

        # 隨機縮放和裁剪
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 裁剪回原始尺寸
            if new_width > self.width or new_height > self.height:
                left = (new_width - self.width) // 2
                top = (new_height - self.height) // 2
                image = image.crop((left, top, left + self.width, top + self.height))
            else:
                # 如果縮小了，創建新圖片並居中貼上
                new_image = Image.new('RGB', (self.width, self.height), (255, 255, 255))
                paste_x = (self.width - new_width) // 2
                paste_y = (self.height - new_height) // 2
                new_image.paste(image, (paste_x, paste_y))
                image = new_image

        return image

    def _apply_filters(self, image: Image.Image) -> Image.Image:
        """
        應用濾鏡效果

        Args:
            image: 原始圖片

        Returns:
            處理後的圖片
        """
        # 隨機應用模糊
        if random.random() < 0.3:
            blur_radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 隨機調整對比度
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)

        # 隨機調整亮度
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)

        return image

    def generate_simple(self, text: Optional[str] = None) -> Tuple[Image.Image, str]:
        """
        生成簡單樣式的 CAPTCHA

        Args:
            text: 指定文字，若為 None 則隨機生成

        Returns:
            (圖片, 文字)
        """
        if text is None:
            text = self.generate_text()

        # 創建圖片
        background_color = random.choice(self.background_colors)
        image = Image.new('RGB', (self.width, self.height), background_color)
        draw = ImageDraw.Draw(image)

        # 獲取字體
        font = self._get_random_font(self.font_size)

        # 計算文字位置
        char_width = self.width // len(text)
        char_height = self.height

        for i, char in enumerate(text):
            # 隨機顏色
            color = random.choice(self.colors)

            # 計算字元位置
            x = i * char_width + random.randint(-5, 5)
            y = (char_height - self.font_size) // 2 + random.randint(-5, 5)

            # 繪製字元
            draw.text((x, y), char, font=font, fill=color)

        # 添加少量干擾
        self._add_noise_dots(draw, (self.width, self.height), count=20)

        return image, text

    def generate_standard(self, text: Optional[str] = None) -> Tuple[Image.Image, str]:
        """
        生成標準樣式的 CAPTCHA

        Args:
            text: 指定文字，若為 None 則隨機生成

        Returns:
            (圖片, 文字)
        """
        if text is None:
            text = self.generate_text()

        # 創建圖片
        background_color = random.choice(self.background_colors)
        image = Image.new('RGB', (self.width, self.height), background_color)
        draw = ImageDraw.Draw(image)

        # 獲取字體
        font_size = random.randint(self.font_size - 4, self.font_size + 4)
        font = self._get_random_font(font_size)

        # 計算文字位置
        char_width = self.width // len(text)

        for i, char in enumerate(text):
            # 隨機顏色
            color = random.choice(self.colors)

            # 隨機位置變化
            x = i * char_width + random.randint(-8, 8)
            y = (self.height - font_size) // 2 + random.randint(-8, 8)

            # 隨機字體大小變化
            char_font_size = font_size + random.randint(-3, 3)
            char_font = self._get_random_font(char_font_size)

            # 繪製字元
            draw.text((x, y), char, font=char_font, fill=color)

        # 添加干擾
        self._add_noise_lines(draw, (self.width, self.height), count=random.randint(2, 4))
        self._add_noise_dots(draw, (self.width, self.height), count=random.randint(30, 60))

        # 輕微扭曲
        image = self._distort_image(image)

        return image, text

    def generate_complex(self, text: Optional[str] = None) -> Tuple[Image.Image, str]:
        """
        生成複雜樣式的 CAPTCHA

        Args:
            text: 指定文字，若為 None 則隨機生成

        Returns:
            (圖片, 文字)
        """
        if text is None:
            text = self.generate_text()

        # 創建圖片
        background_color = random.choice(self.background_colors)
        image = Image.new('RGB', (self.width, self.height), background_color)
        draw = ImageDraw.Draw(image)

        # 獲取字體
        font_size = random.randint(self.font_size - 6, self.font_size + 6)
        font = self._get_random_font(font_size)

        # 計算文字位置
        char_width = self.width // len(text)

        for i, char in enumerate(text):
            # 隨機顏色
            color = random.choice(self.colors)

            # 較大的位置變化
            x = i * char_width + random.randint(-12, 12)
            y = (self.height - font_size) // 2 + random.randint(-12, 12)

            # 隨機字體大小變化
            char_font_size = font_size + random.randint(-5, 5)
            char_font = self._get_random_font(char_font_size)

            # 繪製字元
            draw.text((x, y), char, font=char_font, fill=color)

        # 添加大量干擾
        self._add_noise_lines(draw, (self.width, self.height), count=random.randint(5, 8))
        self._add_noise_dots(draw, (self.width, self.height), count=random.randint(80, 120))

        # 強烈扭曲和濾鏡
        image = self._distort_image(image)
        image = self._apply_filters(image)

        return image, text

    def generate(self,
                 text: Optional[str] = None,
                 style: str = "standard") -> Tuple[Image.Image, str]:
        """
        生成 CAPTCHA 圖片

        Args:
            text: 指定文字，若為 None 則隨機生成
            style: 樣式 ("simple", "standard", "complex")

        Returns:
            (圖片, 文字)
        """
        if style == "simple":
            return self.generate_simple(text)
        elif style == "standard":
            return self.generate_standard(text)
        elif style == "complex":
            return self.generate_complex(text)
        else:
            raise ValueError(f"不支援的樣式: {style}")

    def generate_batch(self,
                       count: int,
                       style: str = "standard",
                       save_dir: Optional[str] = None) -> List[Tuple[Image.Image, str]]:
        """
        批次生成 CAPTCHA 圖片

        Args:
            count: 生成數量
            style: 樣式
            save_dir: 保存目錄（可選）

        Returns:
            圖片和文字的列表
        """
        results = []

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            image, text = self.generate(style=style)
            results.append((image, text))

            if save_dir:
                filename = f"{text}_{i+1:03d}.png"
                image.save(save_path / filename)

        if save_dir:
            logger.info(f"批次生成 {count} 張圖片，保存到: {save_dir}")

        return results

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """
        將圖片轉換為 base64 字串

        Args:
            image: PIL 圖片
            format: 圖片格式

        Returns:
            base64 編碼的圖片字串
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_base64

    def generate_for_api(self,
                         text: Optional[str] = None,
                         style: str = "standard") -> Dict[str, Union[str, float]]:
        """
        為 API 生成 CAPTCHA（返回 base64 格式）

        Args:
            text: 指定文字
            style: 樣式

        Returns:
            包含 base64 圖片和文字的字典
        """
        import time

        start_time = time.time()
        image, generated_text = self.generate(text, style)
        generation_time = time.time() - start_time

        image_base64 = self.image_to_base64(image)

        return {
            "image_base64": image_base64,
            "text": generated_text,
            "style": style,
            "generation_time": generation_time,
            "image_size": f"{self.width}x{self.height}"
        }


def create_generator(config: Optional[Dict] = None) -> CaptchaGenerator:
    """
    創建 CAPTCHA 生成器

    Args:
        config: 配置參數

    Returns:
        生成器實例
    """
    if config is None:
        config = {}

    generator = CaptchaGenerator(
        width=config.get('width', 128),
        height=config.get('height', 64),
        font_size=config.get('font_size', 24),
        character_count=config.get('character_count', 4)
    )

    return generator


def generate_training_data(output_dir: str,
                          count: int = 1000,
                          style: str = "standard",
                          config: Optional[Dict] = None) -> str:
    """
    生成訓練資料

    Args:
        output_dir: 輸出目錄
        count: 生成數量
        style: 樣式
        config: 生成器配置

    Returns:
        輸出目錄路徑
    """
    logger.info(f"開始生成 {count} 張訓練圖片...")

    generator = create_generator(config)
    generator.generate_batch(count, style, output_dir)

    logger.info(f"訓練資料生成完成: {output_dir}")
    return output_dir