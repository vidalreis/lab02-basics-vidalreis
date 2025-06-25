import numpy as np
from skimage.transform import resize

def to_grayscale(img, mode='average'):
    if mode == 'average':
        gray_img = img.mean(axis=2)
    elif mode == 'luminance_perception':
        gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    elif mode == 'linear_approximation':
        gray_img = np.dot(img[..., :3], [0.2126, 0.7152, 0.0722])
    else:
        raise ValueError("Modo inválido. Escolha entre: 'average', 'luminance_perception', 'linear_approximation'.")
    return gray_img.astype(np.uint8)

def upscale_image(img, factor=2, interp='nearest'):
    if factor <= 1:
        raise ValueError("O fator de escala deve ser maior que 1.")
    orig_h, orig_w = img.shape[:2]
    new_h, new_w = int(orig_h * factor), int(orig_w * factor)
    if interp == 'nearest':
        if img.ndim == 2:
            enlarged = np.zeros((new_h, new_w), dtype=img.dtype)
        else:
            enlarged = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
        for y in range(new_h):
            for x in range(new_w):
                src_y = min(int(y / factor), orig_h - 1)
                src_x = min(int(x / factor), orig_w - 1)
                enlarged[y, x] = img[src_y, src_x]
    elif interp == 'bilinear':
        resized = resize(img, (new_h, new_w), order=1, mode='reflect')
        if img.dtype == np.uint8:
            resized = (resized * 255).astype(np.uint8)
        enlarged = resized
    else:
        raise ValueError("Interpolação inválida. Use: 'nearest' ou 'bilinear'.")
    return enlarged

def downscale_image(img, factor=0.5, method='none'):
    if not (0 < factor < 1):
        raise ValueError("O fator de escala deve estar entre 0 e 1.")
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    if method == 'none':
        stride = int(1 / factor)
        reduced = img[::stride, ::stride]
    elif method == 'average':
        reduced = np.zeros((new_h, new_w) + (() if img.ndim == 2 else (3,)), dtype=np.uint8)
        for i in range(new_h):
            for j in range(new_w):
                y_start, y_end = int(i / factor), int((i + 1) / factor)
                x_start, x_end = int(j / factor), int((j + 1) / factor)
                region = img[y_start:y_end, x_start:x_end]
                if img.ndim == 2:
                    reduced[i, j] = np.mean(region)
                else:
                    reduced[i, j] = np.mean(region, axis=(0, 1))
    else:
        raise ValueError("Método inválido. Use: 'none' ou 'average'.")
    return reduced.astype(np.uint8)

def apply_blur(image, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser ímpar.")
    pad = kernel_size // 2
    if image.ndim == 2:
        padded = np.pad(image, pad, mode='reflect')
        blurred = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                blurred[i, j] = np.mean(region)
    else:
        blurred = np.zeros_like(image)
        for c in range(3):
            channel = image[:, :, c]
            padded = np.pad(channel, pad, mode='reflect')
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded[i:i+kernel_size, j:j+kernel_size]
                    blurred[i, j, c] = np.mean(region)
    return blurred.astype(np.float32)
