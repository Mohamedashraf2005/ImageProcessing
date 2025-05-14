from PIL import Image
import numpy as np
import matplotlib as plt
import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog


# ============= Image Processing Functions =============

# 1. Point Operations
def addition(image1, image2):
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return np.clip(image1 + image2, 0, 255).astype(np.uint8)

def subtraction(image1, image2):
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return np.clip(image1 - image2, 0, 255).astype(np.uint8)

def division(image1, image2):
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return np.clip(image1 / (image2 + 1), 0, 255).astype(np.uint8)

def complement(image):
    return np.clip(255 - image, 0, 255).astype(np.uint8) 

# 2. Color Image Operations
def change_red_lighting(image):
    img_copy = image.copy()

     # Ask user for red illumination value
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    try:
        red_value = simpledialog.askinteger(
            "Red Lighting", 
            "Enter red lighting value (e.g., 0 to 255):", 
            minvalue=-255, maxvalue=255
        )
        if red_value is None:
            messagebox.showinfo("Cancelled", "Operation cancelled by user.")
            return img_copy

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")
        return img_copy
    
    # Apply red lighting adjustment
    if len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
        img_copy[:, :, 2] = np.clip(img_copy[:, :, 2] + red_value, 0, 255)
    return img_copy


def swap_r_to_g(image):
    image = Image.fromarray(image).convert("RGB") 
    r, g, b = image.split()   
    new_image = Image.merge("RGB", (g, r, b)) 
    return np.array(new_image)


def eliminate_red(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        result = image.copy()
        result[:, :, 2] = 0
        return result
    else:
        return image

import numpy as np



# 3. Image Histogram Operations
def histogram_stretching(image):
    min_pixel = np.min(image)
    max_pixel = np.max(image)

    # Stretch the histogram
    stretched_image = (image - min_pixel) * (255 / (max_pixel - min_pixel))
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)  # Ensure values are in uint8 range

    return stretched_image

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

def histogram_equalization(image):
    hist_freq,_ = np.histogram(image.flatten(), 256, (0, 256))
    pdf = hist_freq / hist_freq.sum()
    cdf = pdf.cumsum()
    cdf_equalized = (cdf * 255).astype(np.uint8)
    equalized_image = cdf_equalized[image]
    hist_freq_equalized, _ = np.histogram(equalized_image.flatten(), 256, [0, 256])

    return equalized_image, hist_freq, hist_freq_equalized
    

# 4. Neighborhood Processing
#low pass flter == avg filter 
def low_pass_filter(image):
        # Check if the image is RGB (3D array)
    if len(image.shape) == 3:
        # Convert to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
   
    kernel = np.ones((3, 3), np.float32) / 9  

    height, width = image.shape
    filtered_image = np.zeros((height, width), np.float32)
    k_height, k_width = kernel.shape
    k_center_y, k_center_x = k_height // 2, k_width // 2

    for i in range(k_center_y, height - k_center_y):
        for j in range(k_center_x, width - k_center_x):
            neighborhood = image[i - k_center_y:i + k_center_y + 1, j - k_center_x:j + k_center_x + 1]
            filtered_image[i, j] = np.sum(neighborhood * kernel)

    return filtered_image
    

def laplacian_filter(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    filtered = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            filtered[i, j] = np.clip(np.sum(region * kernel), 0, 255)
    return filtered.astype(np.uint8)


def maximum_filter(image):
    filtered = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            filtered[i, j] = np.max(region)
    return filtered.astype(np.uint8)

def minimum_filter(image):
    filtered = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            filtered[i, j] = np.min(region)
    return filtered.astype(np.uint8)

def median_filter(image):
    filtered = np.zeros_like(image)
    rows, cols = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i-1:i+2, j-1:j+2].flatten()
            median_val = sorted(region)[len(region) // 2]
            filtered[i, j] = median_val
    return filtered

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
    return low_pass_filter(image)

#median filter to remove salt and pepper noise 
def salt_pepper_median(image, kernel_size=3):
    return median_filter(image)

def salt_pepper_outlier(image):
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

    image = image.astype(np.float32)
    accumulated = np.zeros_like(image)
    
    for _ in range(num_noisy_images):
        noise = np.zeros_like(image)
        cv2.randn(noise, 0, 20)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        accumulated += noisy_image

    average_image = accumulated / num_noisy_images
    return np.clip(average_image, 0, 255).astype(np.uint8)


def gaussian_average_filter(image):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32)
    kernel /= kernel.sum()  # Normalize to make sum = 1
    filtered = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            filtered[i, j] = np.clip(np.sum(region * kernel), 0, 255)
    return filtered.astype(np.uint8)

# 6. Image Segmentation
def basic_global_thresholding(image):
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image.copy()

    root = tk.Tk()
    root.withdraw()

    threshold_value = simpledialog.askinteger("Threshold Input", "Enter threshold (0–255):", minvalue=0, maxvalue=255)

    if threshold_value is None:
        messagebox.showinfo("Cancelled", "No threshold was entered.")
        return image

    thresholded_image = (img > threshold_value).astype('uint8') * 255
    messagebox.showinfo("Threshold Applied", f"You entered threshold: {threshold_value}")

    return thresholded_image

def automatic_thresholding(image):
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image.copy()

    Theta = np.mean(img)
    done = False

    while not done:
        p1 = img[img >= Theta]
        p2 = img[img < Theta]

        if len(p1) == 0 or len(p2) == 0:
            break

        m1 = np.mean(p1)
        m2 = np.mean(p2)
        Th_next = 0.5 * (m1 + m2)
        done = abs(Theta - Th_next) < 0.5
        Theta = Th_next

    _, im_bw = cv2.threshold(img, Theta, 255, cv2.THRESH_BINARY)
    return im_bw

def adaptive_thresholding(image):
    # Ask user for number of parts
    root = tk.Tk()
    root.withdraw()  # hide main window

    try:
        num_parts = simpledialog.askinteger("Input", "Enter number of horizontal slices (split image from top to bottom):", minvalue=1)
        if num_parts is None:
            messagebox.showinfo("Cancelled", "Operation cancelled by user.")
            return image  # Return original image if cancelled
    except:
        messagebox.showerror("Error", "Invalid input for number of parts.")
        return image

    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image.copy()

    height = img.shape[0]
    step = height // num_parts
    parts = []

    for i in range(num_parts):
        start = i * step
        end = (i + 1) * step if i < num_parts - 1 else height
        part = img[start:end, :]
        _, threshed = cv2.threshold(part, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        parts.append(threshed)

    return np.concatenate(parts, axis=0)

# ==== Sobel Edge Detection ====
def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(sobel_combined)

# ==== Dilation ====
def image_dilation(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    return dilated

# ==== Erosion ====
def image_erosion(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return eroded
