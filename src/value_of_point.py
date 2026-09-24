import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

def true_function(x):
    return np.sin(x) + 0.1 * x**2

def generate_training_data(seed=42, num_samples=500):
    """
    Generate random training data.
    
    Parameters:
    seed (int): Random seed for reproducibility.
    num_samples (int): Number of training samples.

    Returns:
    x_train (np.ndarray): Input training data.
    y_train (np.ndarray): Output training data.
    """
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    x_train = rng.uniform(-10, 10, size=(num_samples, 1))
    y_train = true_function(x_train) + rng.normal(0, 0.05, size=(num_samples, 1))
    return x_train, y_train

def build_model():
    """
    Builds and returns a neural network model.
    
    Returns:
    model (keras.Model): Neural network model.
    """
    model = keras.Sequential([
        keras.Input(shape=(1,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse", metrics=["mae"])
    return model

def train_model(model, x_train, y_train, epochs=100, batch_size=32):
    """
    Trains the model on the training data.
    
    Parameters:
    model (keras.Model): Neural network model.
    x_train (np.ndarray): Input training data.
    y_train (np.ndarray): Output training data.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    history: Training history.
    """
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )
    return history

def evaluate_model(model, x_test):
    """
    Evaluates the model on the test data
    
    Parameters:
    model (keras.Model): Trained neural network model.
    x_test (np.ndarray): Input test data.

    Returns:
    y_pred (np.ndarray): Predicted output
    """
    output = model.predict(x_test, verbose=0)
    return output

def visualize_results(x_train, y_train, x_test, y_pred):
    """Visualize the training data and model predictions."""
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, label="Training Data", alpha=0.3)
    plt.plot(x_test, true_function(x_test), label="True Function", color="green", linestyle="dashed")
    plt.plot(x_test, y_pred, label="NN Prediction", color="red")
    plt.legend()
    plt.title("Function Approximation using Neural Network")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

def predict(model, x_new):
    """
    Predicts the value of a new point.
    
    Parameters:
    model (keras.Model): Trained neural network model.
    x_new (float): New input value.

    Returns:
    predicted_value (float): Predicted output value.
    """
    x = np.array([[x_new]], dtype=np.float32)
    predicted_value = float(model.predict(x, verbose=0)[0, 0])
    return predicted_value

# Main Execution
if __name__ == "__main__":
    # Step 1: Generate Data
    x_train, y_train = generate_training_data()
    x_test = np.linspace(-10, 10, 100).reshape(-1, 1)
    
    # Step 2: Build and Train Model
    model = build_model()
    train_model(model, x_train, y_train)
    
    # Step 3: Evaluate Model
    y_pred = evaluate_model(model, x_test)
    
    # Step 4: Visualize Results
    visualize_results(x_train, y_train, x_test, y_pred)
    
    # Step 5: Predict a new value
    x_new = 7
    y_new_pred = predict(model, x_new)
    print(f"Predicted value at x={x_new}: {y_new_pred:.4f}")
