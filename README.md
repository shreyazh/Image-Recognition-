# WHAT'S THE NUMBER? - An Image Recognition Model

This repository contains a simple deep learning model to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. The model architecture is built using fully connected layers, and the performance is evaluated based on accuracy and confusion matrix visualizations.

## Project Structure

1. **Data Preprocessing**:
   - Loads the MNIST dataset using Keras.
   - Normalizes the pixel values to be between 0 and 1.
   - Reshapes the images to a flattened format (28x28 -> 784).

2. **Model Architectures**:
   - **First Model**: A single dense layer with 10 neurons using a `sigmoid` activation function.
   - **Second Model**: Two dense layers—one with 100 neurons using a `relu` activation function and the second with 10 neurons using `sigmoid`.
   - **Third Model**: Adds a flatten layer as input before the dense layers to handle the raw image shape (28x28).

3. **Training and Evaluation**:
   - Each model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function.
   - The models are trained for 5 epochs and evaluated on test data.
   - Predictions are made and evaluated using confusion matrix and heatmaps.

4. **Confusion Matrix**:
   - Displays the confusion matrix to visually assess the model's performance.

## Requirements

To run this code, you will need the following libraries:

- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

Install them using pip:

```bash
pip install tensorflow numpy pandas matplotlib seaborn
```

## Code Walkthrough

1. **Loading and Preprocessing the Data**:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as nm
import pandas as pd
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)
```

2. **Building the Model**:

```python
structure = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
structure.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
structure.fit(x_train_flattened, y_train, epochs=5)
```

3. **Evaluating and Predicting**:

```python
structure.evaluate(x_test_flattened, y_test)
y_predicted = structure.predict(x_test_flattened)
```

4. **Visualizing the Confusion Matrix**:

```python
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
```

## Model Training

The model is trained using the Adam optimizer and the sparse categorical cross-entropy loss function. After training, we evaluate the model on the test dataset and visualize its performance using confusion matrices.

## Results

Each model is evaluated on its accuracy, and confusion matrices are plotted to show how well the model performs across each digit class.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This README describes the project's purpose, how to run the code, the model's architecture, and visualizations. It also includes installation instructions and code explanations.
