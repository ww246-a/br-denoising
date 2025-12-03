import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import rcParams

def light_intensity_parameters(light_intensity):
    light_intensity = np.array(light_intensity)
    A1 = (2.20E-10) + (1.53E-11) * light_intensity
    t1 = -0.1 * np.exp(-light_intensity / 89.36) + 0.63
    return A1, t1


def load_local_mnist(data_dir=r"D:\Zzj\output\noise_image2"):
    x_train = np.load(f"{data_dir}/x_train_gaussian.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    x_test = np.load(f"{data_dir}/x_test_gaussian.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_local_mnist()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def calculate_gray_values(images):
    gray_values_list = [image.flatten() for image in images]
    return gray_values_list
gray_values_list = calculate_gray_values(x_train)


def calculate_light_intensity(gray_values_list):
    light_intensity_list = []

    for gray_values in gray_values_list:
        light_intensity = gray_values
        light_intensity_list.append(light_intensity)

    return light_intensity_list
light_intensity_list = calculate_light_intensity(gray_values_list)

def calculate_light_intensity_parameters(light_intensity_list):
    all_A1_t1_list = []

    for light_intensity in light_intensity_list:
        A1_t1_values = []
        for li in light_intensity:
            A1, t1 = light_intensity_parameters(li)
            A1_t1_values.append((A1, t1))
        all_A1_t1_list.append(A1_t1_values)

    return all_A1_t1_list
all_A1_t1_list = calculate_light_intensity_parameters(light_intensity_list)


def generate_light_current_functions(all_A1_t1_list):
    light_current_function_list = []

    for A1_t1_values in all_A1_t1_list:
        light_current_functions = []
        for A1, t1 in A1_t1_values:
            light_current_functions.append((A1, t1))
        light_current_function_list.append(light_current_functions)

    return light_current_function_list
light_current_function_list = generate_light_current_functions(all_A1_t1_list)



def calculate_light_current_at_x0(all_A1_t1_list, x=0):
    light_current_list = []

    for A1_t1_values in all_A1_t1_list:
        light_current = []
        for A1, t1 in A1_t1_values:
            light_current_value = A1 * np.exp(-x / t1)
            light_current.append(light_current_value)
        light_current_list.append(light_current)

    return light_current_list
light_current_list = calculate_light_current_at_x0(all_A1_t1_list)

def calculate_grayrate(light_current_list, gray_values_list, all_A1_t1_list):
    recalculated_grayrate_list = []

    for light_current, gray_values, A1_t1_values in zip(light_current_list, gray_values_list, all_A1_t1_list):
        recalculated_grayrate = []
        for lc, gray, (A1, _) in zip(light_current, gray_values, A1_t1_values):
            grayrate_value = (lc * gray / A1) if A1 != 0 else 0
            grayrate_value = np.clip(grayrate_value, 0, 255)
            recalculated_grayrate.append(grayrate_value)
        recalculated_grayrate_list.append(recalculated_grayrate)

    return recalculated_grayrate_list


recalculated_grayrate_list = calculate_grayrate(light_current_list, gray_values_list, all_A1_t1_list)




def reconstruct_gray_images(recalculated_grayrate_list):
    reconstructed_images = [
        np.array(grayrate).reshape(28, 28) for grayrate in recalculated_grayrate_list
    ]
    return reconstructed_images
reconstructed_images = reconstruct_gray_images(recalculated_grayrate_list)


def calculate_light_current_for_x_values(all_A1_t1_list, x_values):
    light_current_dict = {}
    for x in x_values:
        light_current_list = calculate_light_current_at_x0(all_A1_t1_list, x=x)
        light_current_dict[x] = light_current_list
    return light_current_dict


def reconstruct_images_for_x_values(light_current_dict, gray_values_list, all_A1_t1_list):
    reconstructed_images_dict = {}
    for x, light_current_list in light_current_dict.items():
        recalculated_grayrate_list = calculate_grayrate(light_current_list, gray_values_list, all_A1_t1_list)
        reconstructed_images = reconstruct_gray_images(recalculated_grayrate_list)
        reconstructed_images_dict[x] = reconstructed_images
    return reconstructed_images_dict


def preprocess_data(images, labels):
    X = np.array(images).astype('float32') / 255.0
    X = X.reshape((X.shape[0], -1))
    y = labels.flatten()
    return X, y


def build_bp_model(input_shape=784, num_classes=10):
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

gray_values_list = calculate_gray_values(x_train)
light_intensity_list = calculate_light_intensity(gray_values_list)
all_A1_t1_list = calculate_light_intensity_parameters(light_intensity_list)
light_current_function_list = generate_light_current_functions(all_A1_t1_list)
light_current_list = calculate_light_current_at_x0(all_A1_t1_list)
recalculated_grayrate_list = calculate_grayrate(light_current_list, gray_values_list, all_A1_t1_list)
reconstructed_images = reconstruct_gray_images(recalculated_grayrate_list)

X_train, y_train_processed = preprocess_data(reconstructed_images, y_train)
X_test, y_test_processed = preprocess_data(x_test, y_test)

model = build_bp_model()

history = model.fit(X_train, y_train_processed, epochs=30, batch_size=128,
                    validation_data=(X_test, y_test_processed))

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = np.mean(y_pred_classes == y_test_processed)
recall = keras.metrics.Recall()(y_test_processed, y_pred_classes).numpy()
precision = keras.metrics.Precision()(y_test_processed, y_pred_classes).numpy()
mse = mean_squared_error(y_test_processed, y_pred_classes)
conf_matrix = confusion_matrix(y_test_processed, y_pred_classes)

def optimize_x_for_accuracy_exclude_x0(all_A1_t1_list, gray_values_list, y_train, y_test, step=0.02, max_x=5, target_accuracy=0.95):
    best_x = None
    best_accuracy = 0
    accuracy_trend = []
    x_values = []
    x0_accuracy = None
    x0_history = None
    best_history = None

    iteration_data = []

    current_x = 0
    while current_x < max_x:
        light_current_list = calculate_light_current_at_x0(all_A1_t1_list, x=current_x)
        recalculated_grayrate_list = calculate_grayrate(light_current_list, gray_values_list, all_A1_t1_list)
        reconstructed_images = reconstruct_gray_images(recalculated_grayrate_list)

        X_train, y_train_processed = preprocess_data(reconstructed_images, y_train)
        X_test, y_test_processed = preprocess_data(x_test, y_test)

        model = build_bp_model()
        history = model.fit(X_train, y_train_processed, epochs=30, batch_size=128,
                            validation_data=(X_test, y_test_processed), verbose=0)
        val_accuracy = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        train_loss = history.history['loss'][-1]
        print(
            f"x={current_x:.2f}，训练准确率: {train_accuracy:.4f}，验证准确率: {val_accuracy:.4f}，训练损失: {train_loss:.4f}，验证损失: {val_loss:.4f}")

        iteration_data.append({
            'x': current_x,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history.history  
        })

        if current_x == 0:
            x0_accuracy = val_accuracy
            x0_history = history.history

        if current_x != 0 and val_accuracy >= target_accuracy and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_x = current_x
            best_history = history.history

        output_file = f"D:\Zzj\output\curve_new2\BP_bestx_{current_x:.2f}.txt"
        with open(output_file, 'w') as f:
            f.write("Epoch\tTrain_Accuracy\tVal_Accuracy\tTrain_Loss\tVal_Loss\n")
            for epoch in range(len(history.history['accuracy'])):
                train_acc = history.history['accuracy'][epoch]
                val_acc = history.history['val_accuracy'][epoch]
                train_loss = history.history['loss'][epoch]
                val_loss = history.history['val_loss'][epoch]
                f.write(f"{epoch + 1}\t{train_acc:.4f}\t{val_acc:.4f}\t{train_loss:.4f}\t{val_loss:.4f}\n")
        print(f"最佳x={current_x:.2f}的训练日志已保存到: {output_file}")

        accuracy_trend.append(val_accuracy)
        x_values.append(current_x)
        current_x += step

        iteration_output_file = "D:\Zzj\output\curve_new2\BP_x.txt"
        with open(iteration_output_file, 'w') as f:
            f.write("x\tTrain_Accuracy\tVal_Accuracy\tTrain_Loss\tVal_Loss\n")
            for data in iteration_data:
                f.write(
                    f"{data['x']:.2f}\t{data['train_accuracy']:.4f}\t{data['val_accuracy']:.4f}\t{data['train_loss']:.4f}\t{data['val_loss']:.4f}\n")
        print(f"每次迭代x的数据已保存到: {iteration_output_file}")

    return best_x, best_accuracy, x0_accuracy, accuracy_trend, x_values, x0_history, best_history

best_x, best_accuracy, x0_accuracy, accuracy_trend, x_values, x0_history, best_history = optimize_x_for_accuracy_exclude_x0(
    all_A1_t1_list, gray_values_list, y_train, y_test, step=0.02, max_x=5, target_accuracy=0.6
)


print(f"x=0 的准确率: {x0_accuracy:.4f}")
if best_x is not None:
    print(f"最佳 x 值（除 x=0 外）: {best_x}")
    print(f"最佳准确率（除 x=0 外）: {best_accuracy:.4f}")
else:
    print("未找到除 x=0 外准确率大于等于目标值的点。")


plt.figure(figsize=(15, 5))

if best_x is not None:
    plt.subplot(1, 2, 1)
    plt.plot(x0_history['accuracy'], label='accuracy when x=0 ', linestyle='--')
    plt.plot(best_history['accuracy'], label=f'accuracy when x={best_x:.2f} ', linestyle='-')
    plt.plot(x0_history['loss'], label='loss when x=0', linestyle='--')
    plt.plot(best_history['loss'], label=f'loss when x={best_x:.2f} ', linestyle='-')
    plt.title('Comparison of accuracy and loss rate for x=0 and optimal x')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

def save_plot1_data(x0_history, best_history, best_x, filename="D:\Zzj\output\curve_new2\BP_plot1_comparison_data.txt"):
    with open(filename, 'w') as f:
        f.write("Epoch\tx0_Train_Accuracy\tx0_Val_Accuracy\tx0_Train_Loss\tx0_Val_Loss\t")
        f.write(f"bestx_Train_Accuracy\tbestx_Val_Accuracy\tbestx_Train_Loss\tbestx_Val_Loss\n")

        num_epochs = len(x0_history['accuracy'])
        for epoch in range(num_epochs):
            x0_train_acc = x0_history['accuracy'][epoch]
            x0_val_acc = x0_history['val_accuracy'][epoch]
            x0_train_loss = x0_history['loss'][epoch]
            x0_val_loss = x0_history['val_loss'][epoch]

            best_train_acc = best_history['accuracy'][epoch]
            best_val_acc = best_history['val_accuracy'][epoch]
            best_train_loss = best_history['loss'][epoch]
            best_val_loss = best_history['val_loss'][epoch]

            f.write(
                f"{epoch + 1}\t{x0_train_acc:.4f}\t{x0_val_acc:.4f}\t{x0_train_loss:.4f}\t{x0_val_loss:.4f}\t"
                f"{best_train_acc:.4f}\t{best_val_acc:.4f}\t{best_train_loss:.4f}\t{best_val_loss:.4f}\n"
            )
    print(f"图1的对比数据已保存到: {filename}")


if best_x is not None:
    save_plot1_data(x0_history, best_history, best_x)
else:
    print("未找到最佳 x 值，无法保存图1数据。")


plt.subplot(1, 2, 2)
plt.plot(x_values, accuracy_trend, marker='o', label='Accuracy trends')
plt.title('Trend of accuracy with x')
plt.xlabel('x')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
