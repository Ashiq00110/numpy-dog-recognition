import numpy as np
from typing import Tuple, List

class NeuralNetwork:
    """
    A simple neural network built from scratch using NumPy for dog recognition.
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers representing the size of each layer
                        e.g., [784, 128, 64, 2] for input, hidden1, hidden2, output
            learning_rate: Learning rate for backpropagation
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.cache = {}
        
        # Initialize weights and biases
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights with He initialization and biases with zeros."""
        np.random.seed(42)
        
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * \
                np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid."""
        return x * (1 - x)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_features)
        
        Returns:
            Output predictions
        """
        self.cache = {'A0': X}
        A = X
        
        # Forward pass through hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.relu(Z)
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A
        
        # Output layer with Sigmoid (binary classification)
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.sigmoid(Z)
        self.cache[f'Z{len(self.weights)}'] = Z
        self.cache[f'A{len(self.weights)}'] = A
        
        return A
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """
        Backpropagation to compute gradients.
        
        Args:
            X: Input data
            y: True labels
            output: Network output
        """
        m = X.shape[0]
        
        # Output layer gradient (binary cross-entropy)
        dA = -(y / output + (1 - y) / (1 - output)) / m
        dZ = dA * self.sigmoid_derivative(output)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            A_prev = self.cache[f'A{i}']
            
            # Compute gradients
            dW = np.dot(A_prev.T, dZ)
            dB = np.sum(dZ, axis=0, keepdims=True)
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * dB
            
            # Compute gradient for previous layer
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.relu_derivative(self.cache[f'Z{i}'])
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> List[float]:
        """
        Train the neural network.
        
        Args:
            X: Training data of shape (num_samples, num_features)
            y: Training labels of shape (num_samples, 1)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
        
        Returns:
            List of loss values per epoch
        """
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch gradient descent
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, output)
                
                # Compute loss (binary cross-entropy)
                loss = -np.mean(y_batch * np.log(output + 1e-8) + 
                               (1 - y_batch) * np.log(1 - output + 1e-8))
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            threshold: Classification threshold
        
        Returns:
            Binary predictions (0 or 1)
        """
        output = self.forward(X)
        return (output > threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Input data
        
        Returns:
            Probability of being a dog
        """
        return self.forward(X)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate accuracy on a dataset.
        
        Args:
            X: Input data
            y: True labels
            threshold: Classification threshold
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(X, threshold)
        return np.mean(predictions == y)
