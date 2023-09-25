import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 创建一个简单的神经网络模型
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 创建虚拟数据集（示例中的数据是随机生成的，实际应用中需要使用真实数据集）
num_samples = 1000
x_train = np.random.rand(num_samples, 64, 64, 3)
y_train = np.random.randint(2, size=num_samples)

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32)

# 在实际应用中，你需要提供真实的图像数据并进行训练。

# 使用模型进行预测
sample_image = np.random.rand(1, 64, 64, 3)  # 创建一个随机图像
predictions = model.predict(sample_image)

# 输出预测结果
if predictions[0][0] > 0.5:
    print("这是一张狗的图像")
else:
    print("这是一张猫的图像")
