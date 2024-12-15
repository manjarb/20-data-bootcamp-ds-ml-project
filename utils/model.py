import pickle
import os

def save_model(model, filepath):
    """
    Save the trained model using pickle.

    Parameters:
    - model: Trained model object to save.
    - filepath: Path where the model will be saved, e.g., 'model/linear_regression.pkl'.

    Returns:
    - None
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the model
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully to: {filepath}")

def load_model(filepath):
    """
    Load a saved model using pickle.

    Parameters:
    - filepath: Path to the saved model file.

    Returns:
    - Loaded model object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at: {filepath}")
    
    # Load the model
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from: {filepath}")
    return model

# Sample
# Saving Model
# Assume 'model' is your trained model
# save_model(model, 'model/linear_regression_model.pkl')

# Load the model for reuse
# loaded_model = load_model('model/linear_regression_model.pkl')

# Use the loaded model for predictions
# predictions = loaded_model.predict(X_test)
# print("Predictions:", predictions)
