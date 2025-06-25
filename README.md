# Neural-Network-for-House-Price-Prediction

This project demonstrates the development and evaluation of a neural network for predicting median house values in California districts, utilizing the California Housing dataset. The goal was to build a robust regression model capable of generalizing to unseen data, while also understanding key concepts of neural network training.

# Context
Problem: Predicting continuous numerical values (median house prices) is a fundamental regression task in machine learning. Accurate house price prediction can have applications in real estate valuation, market analysis, and investment planning.

Dataset: The project uses the California Housing dataset, sourced from the 1990 U.S. Census. This dataset contains 20,640 instances with 8 numerical features (e.g., median income, house age, average rooms, population, latitude, longitude) and a target variable: MedHouseVal (median house value in hundreds of thousands of dollars). The dataset is clean with no missing values, making it suitable for a beginner project.

Objective: To design, train, and evaluate a neural network model that can predict median house values with reasonable accuracy and generalize well to new, unseen house features.

# Actions
Data Loading & Initial Exploration:

Loaded the California Housing dataset using sklearn.datasets.fetch_california_housing(as_frame=True).

Performed initial exploratory data analysis (EDA) using df.info(), df.describe(), and df.hist() to understand feature distributions and basic statistics. Confirmed no missing values.

Data Preprocessing:

Separated the features (X) from the target variable (y).

Split the dataset into training (80%) and testing (20%) sets using train_test_split with a random_state for reproducibility.

Applied StandardScaler to normalize the numerical features of both the training (X_train_scaled) and testing (X_test_scaled) sets. This is crucial for optimal neural network performance.

Neural Network Architecture Definition (Phase 1):

Constructed a Sequential Keras model with:

An input layer matching the 8 features of the dataset.

Two hidden Dense layers, each with 128 neurons and a relu activation function, to capture complex non-linear relationships.

A single-neuron output Dense layer with a linear activation function, appropriate for predicting a continuous value.

Model Compilation (Phase 2):

Configured the model for training:

Optimizer: Adam, a robust and widely used optimization algorithm.

Loss Function: Mean Squared Error (MSE), the standard choice for regression tasks.

Metrics: Mean Absolute Error (MAE) for easier interpretation of prediction errors in dollar terms.

Model Training (Phase 3):

Trained the model for up to 100 epochs using model.fit().

Included a validation_split of 20% to monitor performance on unseen data during training and detect overfitting.

Implemented EarlyStopping with a patience of 10 epochs (monitoring val_loss) and restore_best_weights=True to automatically halt training when validation performance ceased to improve, ensuring the retention of the best-performing weights.

# Results
The training process demonstrated effective learning and the successful application of early stopping:

Training Convergence: Both training loss (MSE) and training MAE consistently decreased throughout the training process, indicating the model was learning to fit the training data well.

Optimal Generalization with Early Stopping:

Analysis of the validation curves (Loss and MAE) showed that the model achieved its best generalization performance (lowest val_loss and val_MAE) around Epoch 5.

Beyond this point, the validation metrics fluctuated and did not consistently improve, indicating the onset of overfitting.

The EarlyStopping callback successfully intervened, stopping training after approximately 15 epochs (due to 10 epochs of patience after the optimal point at Epoch 5) and restoring the model to its best-performing state.

Final Model Performance on Test Set (Phase 4):

Evaluated the final, optimized model on the completely unseen X_test_scaled data.

Test MAE: ~$35,370 (indicating, on average, predictions were off by about $35,370).

Test MSE Loss: ~0.2823.

These test set results were highly consistent with the validation set performance, confirming the model's ability to generalize effectively to new data.

# Next Steps
Hyperparameter Tuning: Explore different network architectures (e.g., varying the number of neurons in hidden layers, adding more layers) and optimizer learning rates to potentially achieve even lower errors.

Feature Engineering: Investigate creating new features from existing ones (e.g., combining latitude and longitude into regional indicators) that might provide more predictive power.

Advanced Regularization: If significant overfitting persists despite early stopping, consider incorporating more advanced regularization techniques like Dropout layers or L1/L2 regularization to the network architecture.

Error Analysis: Dive deeper into the predictions that had the largest errors to understand why the model struggled with those specific house values. This could reveal data biases or uncaptured relationships.

Deployment (Conceptual): Consider how this model could be deployed as a simple web service to accept new house features and return a predicted price.
