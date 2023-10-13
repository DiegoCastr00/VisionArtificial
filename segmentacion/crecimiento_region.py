import numpy as np
import random
import matplotlib.pyplot as plt

def region_growing(image, threshold):
    height, width = image.shape
    labeled = np.zeros_like(image, dtype=np.uint8)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    num_seeds = 50
    
    def create_seeds(image, num_seeds):
        seeds = []
        height, width = image.shape
        for _ in range(num_seeds):
            random_x = random.randint(0, height - 1)
            random_y = random.randint(0, width - 1)
            seeds.append((random_x, random_y))
        return seeds

    seeds = create_seeds(image, num_seeds=10)
    seed = random.choice(seeds)
    def valid(x, y):
        return 0 <= x < height and 0 <= y < width

    to_process = [seed]
    while to_process:
        x, y = to_process.pop()
        if labeled[x, y] == 0:
            labeled[x, y] = 255
            for neighbor in neighbors:
                nx, ny = x + neighbor[0], y + neighbor[1]
                if valid(nx, ny) and labeled[nx, ny] == 0 and abs(int(image[nx, ny]) - int(image[x, y])) < threshold:
                    to_process.append((nx, ny))
    

    plt.figure(figsize=(8, 8))
    plt.imshow(labeled, cmap='gray')
    plt.title('Region Growing')
    plt.axis('off')
    plt.show()


