import numpy as np
import matplotlib.pyplot as plt

input_data = np.array([[20,30],[23,32],[28,40],[30,44]])

'''
Normalize input_data
x max: 30
x min: 20
y max: 44
y min: 30
(x max) - (x min): 10
(y max) - (y min): 14
'''

input_data_normalized = (input_data - np.array([20, 30])) / np.array([10, 14])
# print(input_data_normalized)

data_number = input_data.shape[0]

epochs = 1000
alpha = 0.1

w0 = 0.1
w1 = 0.1

for epoch in range(epochs):
    dw0 = 0
    dw1 = 0
    for i in range(data_number):
        dw0 = dw0 + 2 * w0 + 2 * w1 * input_data_normalized[i, 0] - 2 * input_data_normalized[i, 1]
        dw1 = dw1 + input_data_normalized[i, 0] * (2 * w1 * input_data_normalized[i, 0] + 2 * w0 - 2 * input_data_normalized[i, 1])
        error = w1 ** 2 + w1 ** 2 * input_data_normalized[i, 0] + input_data_normalized[i, 1] ** 2 - 2 * w1 * input_data_normalized[i, 0] * input_data_normalized[i, 1] - 2 * input_data_normalized[i, 1] * w0

    w0 = w0 - alpha * (dw0)
    w1 = w1 - alpha * (dw1)


x = np.linspace(0, 1, 100)
y = w0 + w1 * x
plt.plot(x, y)
for u in range(data_number):
    plt.scatter(input_data_normalized[u, 0], input_data_normalized[u, 1])
plt.show()