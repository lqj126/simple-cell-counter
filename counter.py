import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_fluorescent_dots(
    image_path,
    threshold_ratio=0.5,
    min_area=5,
    gaussian_kernel=(5, 5),
    debug=False
):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at '{image_path}' could not be loaded.")

    # Apply Gaussian blur (optional)
    blurred = image  # Uncomment the line below if denoising is needed
    # blurred = cv2.GaussianBlur(image, gaussian_kernel, 0)

    # Thresholding based on the maximum pixel intensity
    max_val = np.max(blurred)
    threshold = int(threshold_ratio * max_val)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Convert grayscale image to BGR for drawing
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    count = 0
    for i in range(1, num_labels):  # Skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            count += 1
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # Draw rectangle around the detected dot
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if debug:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Binary Threshold Result")
        plt.imshow(binary, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title(f"Detected Dots: {count}")
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.show()

    return count, output_image

def main():
    image_path = r"path2image\image.jpg"
    output_path = r"path2output\output_with_boxes.jpg"

    count, result_img = count_fluorescent_dots(
        image_path=image_path,
        threshold_ratio=0.288,
        min_area=1,
        debug=True
    )

    print(f"Number of fluorescent dots: {count}")
    cv2.imwrite(output_path, result_img)

if __name__ == "__main__":
    main()
