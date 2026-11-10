"""Bounded patch reads shared by UNI2-h and CellViT preparation."""
from typing import List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def require_package(import_name: str, install_name: Optional[str] = None):
    try:
        return __import__(import_name)
    except ImportError as exc:
        pkg = install_name or import_name
        raise ImportError(
            f"Missing dependency '{pkg}'. Run this script in the same conda "
            "environment used for UNI embedding extraction."
        ) from exc


def normalize_rgb_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)

    while img.ndim > 0 and img.shape[0] == 1:
        img = img[0]

    img = np.squeeze(img)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape after normalization: {img.shape}")

    if img.shape[-1] > 3:
        img = img[..., :3]

    return img


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        img = img.astype(np.float32)
        img = (img - float(info.min)) / max(float(info.max - info.min), 1.0)
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    img = img.astype(np.float32)
    max_value = float(np.nanmax(img)) if img.size else 1.0
    if max_value <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def pad_patch(img: np.ndarray, target_h: int, target_w: int, pad_value: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    out[:h, :w, :] = img[:target_h, :target_w, :]
    return out


class SlideReader:
    backend_name = "base"

    @property
    def dimensions(self) -> Tuple[int, int]:
        raise NotImplementedError

    def read_patch(self, x0: int, y0: int, width: int, height: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenSlideReader(SlideReader):
    backend_name = "openslide"

    def __init__(self, path: str):
        openslide = require_package("openslide", "openslide-python")
        self.slide = openslide.OpenSlide(path)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self.slide.dimensions)

    def read_patch(self, x0: int, y0: int, width: int, height: int) -> np.ndarray:
        patch = self.slide.read_region((int(x0), int(y0)), 0, (int(width), int(height)))
        return np.asarray(patch.convert("RGB"))

    def close(self) -> None:
        self.slide.close()


class TiffZarrReader(SlideReader):
    backend_name = "tiff-zarr"

    def __init__(self, path: str):
        tifffile = require_package("tifffile")
        zarr = require_package("zarr")
        self.tif = tifffile.TiffFile(path)
        self.store = self.tif.series[0].aszarr()
        self.arr = zarr.open(self.store, mode="r")
        shape = tuple(self.arr.shape)
        if len(shape) < 2:
            raise ValueError(f"Unsupported tiff-zarr shape: {shape}")
        self._height, self._width = infer_hw_from_shape(shape)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    def read_patch(self, x0: int, y0: int, width: int, height: int) -> np.ndarray:
        y1 = y0 + height
        x1 = x0 + width
        shape = tuple(self.arr.shape)

        if len(shape) == 2:
            patch = self.arr[y0:y1, x0:x1]
        elif len(shape) == 3 and shape[-1] in (3, 4):
            patch = self.arr[y0:y1, x0:x1, :]
        elif len(shape) == 3 and shape[0] in (3, 4):
            patch = self.arr[:, y0:y1, x0:x1]
        else:
            raise ValueError(f"Unsupported tiff-zarr shape for patch reads: {shape}")

        return to_uint8(normalize_rgb_image(np.asarray(patch)))

    def close(self) -> None:
        if hasattr(self.store, "close"):
            self.store.close()
        self.tif.close()


class TiffMemoryReader(SlideReader):
    backend_name = "tiff-memory"

    def __init__(self, path: str):
        tifffile = require_package("tifffile")
        img = tifffile.imread(path)
        self.img = to_uint8(normalize_rgb_image(img))
        self._height, self._width = self.img.shape[:2]

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    def read_patch(self, x0: int, y0: int, width: int, height: int) -> np.ndarray:
        y1 = y0 + height
        x1 = x0 + width
        return self.img[y0:y1, x0:x1, :]


class TiffMemmapReader(SlideReader):
    backend_name = "tiff-memmap"

    def __init__(self, path: str):
        tifffile = require_package("tifffile")
        img = tifffile.memmap(path)
        self.img = normalize_rgb_image(img)
        self._height, self._width = infer_hw_from_shape(tuple(self.img.shape))

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    def read_patch(self, x0: int, y0: int, width: int, height: int) -> np.ndarray:
        y1 = y0 + height
        x1 = x0 + width
        shape = tuple(self.img.shape)

        if len(shape) == 2:
            patch = self.img[y0:y1, x0:x1]
        elif len(shape) == 3 and shape[-1] in (3, 4):
            patch = self.img[y0:y1, x0:x1, :]
        elif len(shape) == 3 and shape[0] in (3, 4):
            patch = self.img[:, y0:y1, x0:x1]
        else:
            raise ValueError(f"Unsupported tiff-memmap shape for patch reads: {shape}")

        return to_uint8(normalize_rgb_image(np.asarray(patch)))


def infer_hw_from_shape(shape: Sequence[int]) -> Tuple[int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[-1] in (3, 4):
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[0] in (3, 4):
        return int(shape[1]), int(shape[2])
    raise ValueError(f"Cannot infer H/W from image shape: {tuple(shape)}")


def open_reader(path: str, requested: str) -> SlideReader:
    errors: List[str] = []

    def try_reader(cls):
        try:
            return cls(path)
        except Exception as exc:
            errors.append(f"{cls.backend_name}: {exc}")
            return None

    if requested == "openslide":
        reader = try_reader(OpenSlideReader)
    elif requested == "tiff-zarr":
        reader = try_reader(TiffZarrReader)
    elif requested == "tiff-memmap":
        reader = try_reader(TiffMemmapReader)
    elif requested == "tiff-memory":
        reader = try_reader(TiffMemoryReader)
    else:
        reader = try_reader(OpenSlideReader)
        if reader is None:
            reader = try_reader(TiffZarrReader)
        if reader is None:
            reader = try_reader(TiffMemmapReader)
        if reader is None:
            reader = try_reader(TiffMemoryReader)

    if reader is None:
        joined = "\n  ".join(errors)
        raise RuntimeError(f"Could not open slide with reader={requested}. Tried:\n  {joined}")

    return reader


def read_thumbnail(path: str, reader_kind: str, max_side: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    if reader_kind in ("auto", "openslide"):
        try:
            reader = OpenSlideReader(path)
            width, height = reader.dimensions
            scale = min(max_side / float(max(width, height)), 1.0)
            thumb_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            thumb = np.asarray(reader.slide.get_thumbnail(thumb_size).convert("RGB"))
            reader.close()
            return thumb, (height, width)
        except Exception:
            if reader_kind == "openslide":
                raise

    tifffile = require_package("tifffile")
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        full_shape = tuple(series.shape)
        full_hw = infer_hw_from_shape(full_shape)

        if getattr(series, "levels", None):
            levels = list(series.levels)
            smallest = min(levels, key=lambda level: np.prod(infer_hw_from_shape(tuple(level.shape))))
            thumb = smallest.asarray()
        else:
            page = min(tif.pages, key=lambda p: p.imagelength * p.imagewidth)
            thumb = page.asarray()

    thumb = to_uint8(normalize_rgb_image(thumb))
    h, w = thumb.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        thumb = np.asarray(Image.fromarray(thumb).resize(new_size, resample=Image.Resampling.BILINEAR))
    return thumb, full_hw
