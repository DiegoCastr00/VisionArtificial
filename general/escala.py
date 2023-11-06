import cv2
def escala(digit_pixels, zoom_factor):
    height, width = digit_pixels.shape[:2]
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)
    zoomed_image = cv2.resize(digit_pixels, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return zoomed_image