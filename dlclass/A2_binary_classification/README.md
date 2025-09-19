### Step-by-Step Implementation

1. **Import Libraries**:
   First, import the necessary libraries.

   ```python
   import numpy as np
   ```

2. **Sigmoid Function**:
   Define the sigmoid function, which is used in logistic regression.

   ```python
   def sigmoid(z):
       return 1 / (1 + np.exp(-z))
   ```

3. **Logistic Regression Class**:
   Create a logistic regression class that includes weight decay.

   ```python
   class LogisticRegression:
       def __init__(self, learning_rate=0.01, num_iterations=1000, weight_decay=0.01):
           self.learning_rate = learning_rate
           self.num_iterations = num_iterations
           self.weight_decay = weight_decay
           self.weights = None
           self.bias = None

       def fit(self, X, y):
           num_samples, num_features = X.shape
           self.weights = np.zeros(num_features)
           self.bias = 0

           for _ in range(self.num_iterations):
               linear_model = np.dot(X, self.weights) + self.bias
               y_predicted = sigmoid(linear_model)

               # Compute gradients
               dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y)) + self.weight_decay * self.weights
               db = (1 / num_samples) * np.sum(y_predicted - y)

               # Update weights and bias
               self.weights -= self.learning_rate * dw
               self.bias -= self.learning_rate * db

       def predict(self, X):
           linear_model = np.dot(X, self.weights) + self.bias
           y_predicted = sigmoid(linear_model)
           y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
           return np.array(y_predicted_class
   ```

4. **Usage Example**:
   You can use the `LogisticRegression` class to fit a model and make predictions.

   ```python
   if __name__ == "__main__":
       # Example dataset
       X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
       y = np.array([0, 1, 1, 0])  # Example labels (XOR problem)

       # Create and train the model
       model = LogisticRegression(learning_rate=0.1, num_iterations=1000, weight_decay=0.01)
       model.fit(X, y)

       # Make predictions
       predictions = model.predict(X)
       print("Predictions:", predictions)
   ```

### Explanation:
- **Weight Decay**: The term `self.weight_decay * self.weights` in the gradient calculation adds the L2 penalty to the weight updates. This discourages large weights, which helps to reduce overfitting.
- **Learning Rate and Iterations**: You can adjust the learning rate and the number of iterations based on your dataset and convergence needs.
- **Prediction**: The `predict` method uses the learned weights and bias to make predictions on new data.

### Conclusion:
This implementation provides a basic logistic regression model with weight decay. You can further enhance it by adding features like early stopping, cross-validation, or more advanced optimization techniques.