from PIL import Image
import numpy as np
import matplotlib as plt
import cv2



# ============= Image Processing Functions =============

# 1. Point Operations
def addition(image1, image2):
    """Addition: y = image1 + image2"""
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return np.clip(image1 + image2, 0, 255).astype(np.uint8)

def subtraction(image1, image2):
    """Subtraction: y = image1 - image2"""
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return np.clip(image1 - image2, 0, 255).astype(np.uint8)

def division(image1, image2):
    """Division: y = image1 / (image2 + 1)"""
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return np.clip(image1 / (image2 + 1), 0, 255).astype(np.uint8)

def complement(image):
    """Complement: y = 255 - image1"""
    return np.clip(255 - image, 0, 255).astype(np.uint8) 

# 2. Color Image Operations
def change_red_lighting(image):
    """Increase the red channel brightness"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        result = image.copy()
        result[:, :, 2] = np.clip(result[:, :, 2] + 50, 0, 255)
        return result
    else:
        return image


def swap_r_to_g(image):
    """Swap R channel to G channel"""
    img = Image.fromarray(image).convert("RGB") 
    r, g, b = img.split()   
    new_img = Image.merge("RGB", (g, r, b)) 
    return np.array(new_img)


def eliminate_red(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        result = image.copy()
        result[:, :, 2] = 0
        return result
    else:
        return image

import numpy as np



# 3. Image Histogram Operations
def histogram_stretching_gray(image):
    """Histogram stretching for grayscale image"""
    RR = image.copy()
    low=np.min(RR)
    high=np.max(RR)
    correct=((RR-low)/(high-low)*255)
    return correct

## for colored image

# def histogram_stretching_gray(image):
#     """Histogram stretching for grayscale or RGB image"""
    
#     # لو الصورة 2D (يعني grayscale)
#     if len(image.shape) == 2:
#         low = np.min(image)
#         high = np.max(image)
#         if high != low:
#             stretched = ((image - low) / (high - low)) * 255
#         else:
#             stretched = image
#         return stretched.astype(np.uint8)

#     # لو الصورة 3D (يعني RGB)
#     elif len(image.shape) == 3 and image.shape[2] == 3:
#         stretched = np.zeros_like(image, dtype=np.float32)
#         for c in range(3):  # لكل قناة R,G,B
#             channel = image[:, :, c]
#             low = np.min(channel)
#             high = np.max(channel)
#             if high != low:
#                 stretched[:, :, c] = ((channel - low) / (high - low)) * 255
#             else:
#                 stretched[:, :, c] = channel
#         return stretched.astype(np.uint8)
    
#     else:
#         raise ValueError("Unsupported image format")

def histogram_equalization_gray(image):
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    hist_orig = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
    img_prop = hist_orig / gray_img.size
    cum_sum_arr = np.cumsum(img_prop)
    equalization_map = np.round(cum_sum_arr * 255).astype('uint8')
    equalized_img = equalization_map[gray_img]

    return equalized_img

# 4. Neighborhood Processing
def average_filter(image):

    kernel = np.ones((3, 3), np.float32) / 9
    filtered = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            filtered[i, j] = np.clip(np.sum(region * kernel), 0, 255)
    return filtered.astype(np.uint8)

def laplacian_filter(image):
    return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)

def maximum_filter(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel)

def minimum_filter(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel)

def median_filter(image):
    return cv2.medianBlur(image, 3)

# def median_filter(image):
#     height, width = image.shape
#     output = np.zeros((height, width), dtype=np.uint8)

#     for i in range(1, height - 1):
#         for j in range(1, width - 1):
#             neighbors = []
#             for x in range(-1, 2):
#                 for y in range(-1, 2):
#                     neighbors.append(image[i + x, j + y])
#             neighbors.sort()
#             median = neighbors[len(neighbors) // 2]
#             output[i, j] = median

#     return output

def mode_filter(image):
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+3, j:j+3].flatten()
            counts = np.bincount(window)
            result[i, j] = np.argmax(counts)
    
    return result

# 5. Image Restoration
def salt_pepper_avg(image):
    """Salt and pepper noise removal using average filter"""
    return average_filter(image)

def salt_pepper_median(image):
    """Salt and pepper noise removal using median filter"""
    filtered = np.zeros_like(image)
    rows, cols = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i-1:i+2, j-1:j+2].flatten()
            median_val = sorted(region)[len(region) // 2]
            filtered[i, j] = median_val
    return filtered


def salt_pepper_outlier(image):
    """Salt and pepper noise removal using outlier method"""
    output = np.zeros_like(image)
    kernel = np.array([[1/8, 1/8, 1/8],
                       [1/8, 0, 1/8],
                       [1/8, 1/8, 1/8]])
    average = cv2.filter2D(image, -1, kernel)
    diff = abs(image.astype(np.int16) - average.astype(np.int16))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if diff[i][j] > 4:
                output[i][j] = average[i][j]
            else:
                output[i][j] = image[i][j]
    return output

def gaussian_image_averaging(image, num_noisy_images=10):
    """Gaussian noise removal using image averaging"""
    image = image.astype(np.float32)
    accumulated = np.zeros_like(image)
    
    for _ in range(num_noisy_images):
        noise = np.zeros_like(image)
        cv2.randn(noise, 0, 20)
        noisy_img = image + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        accumulated += noisy_img

    average_img = accumulated / num_noisy_images
    return np.clip(average_img, 0, 255).astype(np.uint8)


def gaussian_average_filter(image):
    """Gaussian noise removal using average filter"""
    return average_filter(image)

# 6. Image Segmentation
def basic_global_thresholding(image, threshold=127):
    """Basic global thresholding"""
    output = np.zeros_like(image)
    output[image > threshold] = 255
    return output

def automatic_thresholding(image):
    """Automatic thresholding using Otsu's method"""
    _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

def adaptive_thresholding(image):
    """Adaptive thresholding"""
    result = cv2.adaptiveThreshold(image, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   11, 2)
    return result