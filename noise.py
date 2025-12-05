import numpy as np
import keras



save_dir = r"D:\Zzj\output\noise_image"

def add_gaussian_noise(images, mean=0, std=25):

    noise = np.random.normal(mean, std, images.shape)
    noisy_images = np.clip(images + noise, 0, 250).astype(np.uint8)
    return noisy_images


#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



x_train_gaussian = add_gaussian_noise(x_train, mean=0, std=0)


x_test_gaussian = add_gaussian_noise(x_test, mean=-70, std=70)


np.save(f"{save_dir}/x_train_gaussian.npy", x_train_gaussian)
np.save(f"{save_dir}/y_train.npy", y_train)
np.save(f"{save_dir}/x_test_gaussian.npy", x_test_gaussian)
np.save(f"{save_dir}/y_test.npy", y_test)

print(f"The noisy dataset has been saved to the directory: {save_dir}")

import matplotlib.pyplot as plt

def show_images(images, titles, n=5):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.show()


show_images(x_train[:5], ["Original"] * 5)
show_images(x_train_gaussian[:5], ["Gaussian Noise"] * 5)
show_images(x_test_gaussian[:5], ["Gaussian Noise"] * 5)
