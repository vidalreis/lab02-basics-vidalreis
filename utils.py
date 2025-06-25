import numpy as np
from skimage.transform import resize

def to_grayscale(img, mode='average'):
    """
    Converte uma imagem colorida para escala de cinza.
    Modos disponíveis: 'average', 'luminance_perception', 'linear_approximation'.
    """
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
    """
    Aumenta o tamanho de uma imagem usando interpolação por vizinho mais próximo ou bilinear.
    """
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
    """
    Reduz o tamanho da imagem com ou sem filtro de média.
    """
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

                region = img[y_start:y_end, x_start:x_end]  # (h, w) or (h, w, c)
                if img.ndim == 2:
                    reduced[i, j] = np.mean(region)
                else:
                    reduced[i, j] = np.mean(region, axis=(0, 1))

    else:
        raise ValueError("Método inválido. Use: 'none' ou 'average'.")

    return reduced.astype(np.uint8)
