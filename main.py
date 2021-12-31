import numpy as np
import matplotlib.pyplot as plt

input_data = np.array([[20,40],[23,32],[28,40],[30,44]])
data_number = input_data.shape[0]

epochs = 1000
alpha = 0.00005

w0 = 0.1
w1 = 0.1

for epoch in range(epochs):
    dw0 = 0
    dw1 = 0
    for i in range(data_number):
        dw0 = dw0 + 2 * w0 + 2 * w1 * input_data[i, 0] - 2 * input_data[i, 1]
        dw1 = dw1 + input_data[i, 0] * (2 * w1 * input_data[i, 0] + 2 * w0 - 2 * input_data[i, 1])
        error = w1 ** 2 + w1 ** 2 * input_data[i, 0] + input_data[i, 1] ** 2 - 2 * w1 * input_data[i, 0] * input_data[i, 1] - 2 * input_data[i, 1] * w0

    w0 = w0 - alpha * (dw0)
    w1 = w1 - alpha * (dw1)

x = np.linspace(15, 35, 100)
y = w0 + w1 * x
plt.plot(x, y)
for u in range(data_number):
    plt.scatter(input_data[u, 0], input_data[u, 1])
plt.show()