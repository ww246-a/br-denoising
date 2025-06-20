import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Step0: 定义函数 light_intensity_parameters() 确定光强对应的光电流函数系数
def light_intensity_parameters(light_intensity):
    light_intensity = np.array(light_intensity)
    #A1 = (-0.22) + (0.01) * light_intensity #归一化后的曲线
    #t1 = -7.42 * 10**(-4) + 1.04 * (1-np.exp(-(light_intensity-23.75)/70.06))
    A1 = (-2.20E-10) + (1.53E-11) * light_intensity
    t1 = -0.01 * np.exp(-light_intensity / 89.36) + 0.63
    return A1, t1


# Step1: 导入MNIST数据集
def load_local_mnist(data_dir=r"D:\Zzj\output\noise_image2"):
    x_train = np.load(f"{data_dir}/x_train_gaussian.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    x_test = np.load(f"{data_dir}/x_test_gaussian.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    return (x_train, y_train), (x_test, y_test)

# 加载本地数据
(x_train, y_train), (x_test, y_test) = load_local_mnist()

# 类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Step2:计算每个像素的灰度值
def calculate_gray_values(images):
    gray_values_list = [image.flatten() for image in images]  # 展平每张图像
    return gray_values_list
# 调用函数计算灰度值数组
gray_values_list = calculate_gray_values(x_train)


# Step3:灰度值对应光强值
def calculate_light_intensity(gray_values_list):
    light_intensity_list = []

    for gray_values in gray_values_list:  # 遍历每张图像的灰度值数组
        # 计算光强值
        light_intensity = gray_values
        light_intensity_list.append(light_intensity)

    return light_intensity_list
# 调用函数计算光强值数组
light_intensity_list = calculate_light_intensity(gray_values_list)

#
# Step4:计算每个像素的光强参数A、t
def calculate_light_intensity_parameters(light_intensity_list):
    all_A1_t1_list = []

    for light_intensity in light_intensity_list:  # 遍历每张图像的光强值数组
        A1_t1_values = []  # 存储当前图像的 A1 和 t1 参数值
        for li in light_intensity:  # 遍历每个像素的光强值
            A1, t1 = light_intensity_parameters(li)  # 调用 Step0 的函数
            A1_t1_values.append((A1, t1))
        all_A1_t1_list.append(A1_t1_values)

    return all_A1_t1_list
# 调用函数计算 A1 和 t1 参数
all_A1_t1_list = calculate_light_intensity_parameters(light_intensity_list)


# Step5:得到每个像素的光强参数A、t对应的光电流函数
def generate_light_current_functions(all_A1_t1_list):
    light_current_function_list = []

    for A1_t1_values in all_A1_t1_list:  # 遍历每张图像的 A1 和 t1 参数
        light_current_functions = []  # 当前图像的光电流函数
        for A1, t1 in A1_t1_values:  # 遍历每个像素的 A1 和 t1 参数
            # 生成光电流函数形式
            light_current_functions.append((A1, t1))
        light_current_function_list.append(light_current_functions)

    return light_current_function_list
# 调用函数生成光电流函数形式
light_current_function_list = generate_light_current_functions(all_A1_t1_list)



# Step6: 给光电流公式赋初始值x=0并计算
def calculate_light_current_at_x0(all_A1_t1_list, x=0):
    light_current_list = []

    for A1_t1_values in all_A1_t1_list:  # 遍历每张图像的 A1 和 t1 参数
        light_current = []  # 当前图像的光电流值
        for A1, t1 in A1_t1_values:  # 遍历每个像素的 A1 和 t1 参数
            # 计算光电流值
            light_current_value = A1 * np.exp(-x / t1)
            light_current.append(light_current_value)
        light_current_list.append(light_current)

    return light_current_list
# 调用函数计算 x=0 时的光电流值
light_current_list = calculate_light_current_at_x0(all_A1_t1_list)


#
# Step7: 根据光电流值计算对应回的灰度值
def calculate_grayrate(light_current_list, gray_values_list, all_A1_t1_list):
    recalculated_grayrate_list = []

    for light_current, gray_values, A1_t1_values in zip(light_current_list, gray_values_list, all_A1_t1_list):
        recalculated_grayrate = []  # 当前图像重新计算的灰度值
        for lc, gray, (A1, _) in zip(light_current, gray_values, A1_t1_values):
            # 使用公式计算灰度值，并四舍五入后裁剪到 [0, 255]
            grayrate_value = (lc * gray / A1) if A1 != 0 else 0
            grayrate_value = np.clip(grayrate_value, 0, 255)  # 限制范围
            recalculated_grayrate.append(grayrate_value)
        recalculated_grayrate_list.append(recalculated_grayrate)

    return recalculated_grayrate_list


# 调用函数计算重新对应回的灰度值
recalculated_grayrate_list = calculate_grayrate(light_current_list, gray_values_list, all_A1_t1_list)




# Step8:处理图像，并输出原图像和处理过后的图像。
def reconstruct_gray_images(recalculated_grayrate_list):
    reconstructed_images = [
        np.array(grayrate).reshape(28, 28) for grayrate in recalculated_grayrate_list
    ]
    return reconstructed_images
# 调用函数重构灰度图像
reconstructed_images = reconstruct_gray_images(recalculated_grayrate_list)






# Step6: 计算不同 x 值下的光电流值
def calculate_light_current_for_x_values(all_A1_t1_list, x_values):
    """
    为多个 x 值计算光电流列表。
    """
    light_current_dict = {}
    for x in x_values:
        light_current_list = calculate_light_current_at_x0(all_A1_t1_list, x=x)
        light_current_dict[x] = light_current_list
    return light_current_dict

# Step7: 计算不同 x 值下的灰度图像
def reconstruct_images_for_x_values(light_current_dict, gray_values_list, all_A1_t1_list):
    """
    根据不同 x 值的光电流重新生成灰度图像。
    """
    reconstructed_images_dict = {}
    for x, light_current_list in light_current_dict.items():
        recalculated_grayrate_list = calculate_grayrate(light_current_list, gray_values_list, all_A1_t1_list)
        reconstructed_images = reconstruct_gray_images(recalculated_grayrate_list)
        reconstructed_images_dict[x] = reconstructed_images
    return reconstructed_images_dict

# 数据预处理
def preprocess_data(images, labels):
    """
    预处理图像数据，归一化并调整形状。
    """
    X = np.array(images).reshape(-1, 28, 28, 1).astype('float32') / 255.0  # 归一化
    y = labels.flatten()  # 展平标签
    return X, y




# Step9-1：构建 ANN 模型
def build_ann_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),  # 将28x28图像展平为784维向量
        keras.layers.Dense(512, activation='relu'),  # 第一个全连接层
        keras.layers.Dropout(0.4),  # 防止过拟合
        keras.layers.Dense(256, activation='relu'),  # 第二个全连接层
        keras.layers.Dropout(0.4),
        keras.layers.Dense(num_classes, activation='softmax')  # 输出层
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 数据预处理保持不变
def preprocess_data(images, labels):
    X = np.array(images).astype('float32') / 255.0  # 归一化
    X = X[..., np.newaxis]  # 添加通道维度 (28, 28) -> (28, 28, 1)
    y = labels.flatten()  # 将标签展平
    return X, y


# 预处理训练集和测试集
X_train, y_train_processed = preprocess_data(reconstructed_images, y_train)
X_test, y_test_processed = preprocess_data(x_test, y_test)

# 构建ANN模型
input_shape = (28, 28, 1)  # 输入形状保持不变
num_classes = 10  # MNIST的类别数量
model = build_ann_model(input_shape, num_classes)

# 训练模型
history = model.fit(X_train, y_train_processed,
                    epochs=20,  # 增加epochs数量
                    batch_size=64,
                    validation_data=(X_test, y_test_processed))

# 评估模型性能
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 计算分类性能指标
accuracy = np.mean(y_pred_classes == y_test_processed)
recall = keras.metrics.Recall()(y_test_processed, y_pred_classes).numpy()
precision = keras.metrics.Precision()(y_test_processed, y_pred_classes).numpy()
mse = mean_squared_error(y_test_processed, y_pred_classes)
conf_matrix = confusion_matrix(y_test_processed, y_pred_classes)


# 优化x值的函数也需要更新模型构建部分
def optimize_x_for_accuracy_exclude_x0(all_A1_t1_list, gray_values_list, y_train, y_test, step=0.02, max_x=5,
                                       target_accuracy=0.95):
    best_x = None
    best_accuracy = 0
    accuracy_trend = []
    x_values = []
    x0_accuracy = None
    x0_history = None
    best_history = None

    # 新增：记录每次迭代的数据
    iteration_data = []  # 存储每次迭代的数据

    current_x = 0
    while current_x < max_x:
        # 计算光电流和重构图像 (保持不变)
        light_current_list = calculate_light_current_at_x0(all_A1_t1_list, x=current_x)
        recalculated_grayrate_list = calculate_grayrate(light_current_list, gray_values_list, all_A1_t1_list)
        reconstructed_images = reconstruct_gray_images(recalculated_grayrate_list)

        # 数据预处理
        X_train, y_train_processed = preprocess_data(reconstructed_images, y_train)
        X_test, y_test_processed = preprocess_data(x_test, y_test)

        # 使用ANN模型代替CNN
        input_shape = (28, 28, 1)
        num_classes = 10
        model = build_ann_model(input_shape, num_classes)
        history = model.fit(X_train, y_train_processed,
                            epochs=20,  # 增加epochs
                            batch_size=64,
                            validation_data=(X_test, y_test_processed),
                            verbose=0)

        val_accuracy = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        train_loss = history.history['loss'][-1]
        print(
            f"x={current_x:.2f}，训练准确率: {train_accuracy:.4f}，验证准确率: {val_accuracy:.4f}，训练损失: {train_loss:.4f}，验证损失: {val_loss:.4f}")

        # 记录当前x的训练数据
        iteration_data.append({
            'x': current_x,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history.history  # 存储完整训练历史（可选）
        })


        if current_x == 0:
            x0_accuracy = val_accuracy
            x0_history = history.history

        if current_x != 0 and val_accuracy >= target_accuracy and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_x = current_x
            best_history = history.history

        # 保存最佳x的训练日志到txt文件
        output_file = f"D:\Zzj\output\curve_new2\Ann_bestx_{current_x:.2f}.txt"
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

        # 新增：保存每次迭代的数据到txt文件
        iteration_output_file = "D:\Zzj\output\curve_new2\Ann_x.txt"
        with open(iteration_output_file, 'w') as f:
            f.write("x\tTrain_Accuracy\tVal_Accuracy\tTrain_Loss\tVal_Loss\n")
            for data in iteration_data:
                f.write(
                    f"{data['x']:.2f}\t{data['train_accuracy']:.4f}\t{data['val_accuracy']:.4f}\t{data['train_loss']:.4f}\t{data['val_loss']:.4f}\n")
        print(f"每次迭代x的数据已保存到: {iteration_output_file}")

    return best_x, best_accuracy, x0_accuracy, accuracy_trend, x_values, x0_history, best_history

# 调用优化函数
best_x, best_accuracy, x0_accuracy, accuracy_trend, x_values, x0_history, best_history = optimize_x_for_accuracy_exclude_x0(
    all_A1_t1_list, gray_values_list, y_train, y_test, step=0.02, max_x=5, target_accuracy=0.6
)

# 打印结果
print(f"x=0 的准确率: {x0_accuracy:.4f}")
if best_x is not None:
    print(f"最佳 x 值（除 x=0 外）: {best_x}")
    print(f"最佳准确率（除 x=0 外）: {best_accuracy:.4f}")
else:
    print("未找到除 x=0 外准确率大于等于目标值的点。")

# 绘制结果
plt.figure(figsize=(15, 5))

# 图1: x=0 和最佳 x 的准确率和损失率对比
if best_x is not None:
    plt.subplot(1, 2, 1)
    plt.plot(x0_history['accuracy'], label='accuracy when x=0 ', linestyle='--')
    plt.plot(best_history['accuracy'], label=f'accuracy when x={best_x:.2f} ', linestyle='-')
    plt.plot(x0_history['loss'], label='loss when x=0', linestyle='--')
    plt.plot(best_history['loss'], label=f'loss when x={best_x:.2f} ', linestyle='-')
    plt.title('Comparison of accuracy and loss rate for x=0 and optimal x')
    plt.xlabel('Iteration Number')
    plt.ylabel('值')
    plt.legend()

# 保存图1的数据到 txt 文件
def save_plot1_data(x0_history, best_history, best_x, filename="D:\Zzj\output\curve_new2\Ann_plot1_comparison_data.txt"):
    with open(filename, 'w') as f:
        # 写入表头
        f.write("Epoch\tx0_Train_Accuracy\tx0_Val_Accuracy\tx0_Train_Loss\tx0_Val_Loss\t")
        f.write(f"bestx_Train_Accuracy\tbestx_Val_Accuracy\tbestx_Train_Loss\tbestx_Val_Loss\n")

        # 确保 x0_history 和 best_history 的 epoch 数相同
        num_epochs = len(x0_history['accuracy'])
        for epoch in range(num_epochs):
            # x=0 的数据
            x0_train_acc = x0_history['accuracy'][epoch]
            x0_val_acc = x0_history['val_accuracy'][epoch]
            x0_train_loss = x0_history['loss'][epoch]
            x0_val_loss = x0_history['val_loss'][epoch]

            # 最佳 x 的数据
            best_train_acc = best_history['accuracy'][epoch]
            best_val_acc = best_history['val_accuracy'][epoch]
            best_train_loss = best_history['loss'][epoch]
            best_val_loss = best_history['val_loss'][epoch]

            # 写入一行数据
            f.write(
                f"{epoch + 1}\t{x0_train_acc:.4f}\t{x0_val_acc:.4f}\t{x0_train_loss:.4f}\t{x0_val_loss:.4f}\t"
                f"{best_train_acc:.4f}\t{best_val_acc:.4f}\t{best_train_loss:.4f}\t{best_val_loss:.4f}\n"
            )
    print(f"图1的对比数据已保存到: {filename}")

# 调用函数保存数据
if best_x is not None:
    save_plot1_data(x0_history, best_history, best_x)
else:
    print("未找到最佳 x 值，无法保存图1数据。")


# 图2: 随 x 迭代变化的准确率趋势
plt.subplot(1, 2, 2)
plt.plot(x_values, accuracy_trend, marker='o', label='Accuracy trends')
plt.title('Trend of accuracy with x')
plt.xlabel('x')
plt.ylabel('accuracy')
plt.legend()

plt.tight_layout()
plt.show()






