from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten, Reshape
import numpy as np

def create_nn3_model(input_shape, output_shape=1):
    """
    Create the NN3 model for evaluating the correctness of predictions.

    Parameters:
    input_shape (tuple): Shape of the input (e.g., number of predicted text elements).
    output_shape (int): Dimension of the output (default to 1 for binary classification).

    Returns:
    model (Sequential): NN3 model.
    """

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))  # Binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def create_nn2_model(input_shape, output_shape):
    """
    Create the NN2 model for text prediction based on patterns.

    Parameters:
    input_shape (tuple): Shape of the input (e.g., number of patterns).
    output_shape (int): Dimension of the output (e.g., number of predicted text elements).

    Returns:
    model (Sequential): NN2 model.
    """

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(Reshape((1, 128))) 
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_shape, activation='softplus'))  # Adjust activation based on the problem

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model

def create_nn1_model(input_shape, output_shape):
    """
    Create the NN1 model for pattern extraction.

    Parameters:
    input_shape (tuple): Shape of the input (e.g., number of features).
    output_shape (int): Dimension of the output (e.g., number of patterns).

    Returns:
    model (Sequential): NN1 model.
    """

    model = Sequential()
    model.add(Embedding(input_dim=input_shape[0], output_dim=128, input_length=input_shape[1]))
    model.add(LSTM(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def preprocess(texts, max_words=10000, max_sequence_length=100):
    # Create a tokenizer and fit it on the text
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    
    # Convert text to sequences of numbers (tokens)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences to have a consistent length
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    
    return padded_sequences, tokenizer


# Pseudocode for training NN1
def train_nn1(input_text, expected_patterns, input_shape, output_shape, num_epochs=100):
    # Preprocess input text (tokenization, etc.)
    processed_input = input_text
    
    # Train NN1 to predict patterns
    nn1_model = create_nn1_model(input_shape=input_shape, output_shape=output_shape)
    nn1_model.compile(optimizer='adam', loss='mean_squared_error')
    nn1_model.fit(processed_input, expected_patterns, epochs=num_epochs)
    
    return nn1_model


# Pseudocode for using NN1 to extract patterns
def extract_patterns(nn1_model, input_text):
    # Preprocess input text (tokenization, etc.)
    processed_input = input_text
    
    # Use NN1 to predict patterns
    predicted_patterns = nn1_model.predict(processed_input)
    
    return predicted_patterns

# Pseudocode for training NN2
def train_nn2(input_patterns, expected_text, input_shape, output_shape, num_epochs=100):
    # Train NN2 to predict text based on patterns
    print("input_patterns = ", input_patterns)
    print("expected: ", expected_text)
    nn2_model = create_nn2_model(input_shape=input_shape, output_shape=output_shape)
    nn2_model.compile(optimizer='adam', loss='categorical_crossentropy')
    nn2_model.fit(input_patterns, expected_text, epochs=num_epochs)
    
    return nn2_model


# Pseudocode for using NN2 to predict text
def predict_text(nn2_model, input_patterns):
    # Use NN2 to predict text
    predicted_text = nn2_model.predict(input_patterns)
    
    return predicted_text

def train_nn3(predicted_text, expected_text, input_shape, output_shape=1, num_epochs=10):
    # Preprocess predicted and expected text (tokenization, etc.)
    processed_predicted_text = predicted_text
    processed_expected_text = expected_text
    
    # Train NN3 to evaluate the correctness of NN2's predictions
    nn3_model = create_nn3_model(input_shape=input_shape, output_shape=output_shape)
    nn3_model.compile(optimizer='adam', loss='binary_crossentropy')
    nn3_model.fit(processed_predicted_text, processed_expected_text, epochs=num_epochs)
    
    return nn3_model


# Pseudocode for using NN3 to evaluate correctness
def evaluate_correctness(nn3_model, predicted_text, expected_text):
    # Preprocess predicted and expected text (tokenization, etc.)
    
    # Use NN3 to evaluate correctness
    correctness = nn3_model.evaluate(predicted_text, expected_text)
    
    return correctness
def reverse_preprocess(padded_sequences, tokenizer):
    # Reverse padding to get original sequences
    sequences = unpad_sequences(padded_sequences)

    # Convert sequences of numbers back to text
    reversed_texts = tokenizer.sequences_to_texts(sequences)

    return reversed_texts

def unpad_sequences(padded_sequences):
    # Remove padding (zeros) from sequences
    unpad_sequences = []
    for sequence in padded_sequences:
        unpad_sequence = [token for token in sequence if token != 0]
        unpad_sequences.append(unpad_sequence)
    return unpad_sequences

def scale_to_integers(data, comparison_data):
    max_comparison_value = np.max(comparison_data)
    scaling_factor = max_comparison_value / np.max(data)
    scaled_data = np.round(data * scaling_factor).astype(int)
    return scaled_data

# Example input text and expected patterns
input_text = ['I can form complete sentences', 'None of these words are in the other string of words', 'In this list']
processedinput, tokenizer = preprocess(input_text)
mask = processedinput != 0
processedinput = np.ma.masked_array(processedinput, mask=~mask)
reversed_texts = reverse_preprocess(processedinput, tokenizer)
expected_patterns = processedinput
print(expected_patterns)

# Define input and output shapes for the models
input_shape_nn1 = (10000, 100)  # Adjust as needed
output_shape_nn1 = expected_patterns.shape[1]
input_shape_nn2 = expected_patterns.shape[1]
output_shape_nn2 = processedinput.shape[1]  # Adjust as needed
input_shape_nn3 = output_shape_nn2
output_shape_nn3 = output_shape_nn2
# Train NN1
nn1_model = train_nn1(processedinput, expected_patterns, input_shape_nn1, output_shape_nn1)
# Extract patterns using NN1

patterns = extract_patterns(nn1_model, processedinput)
patterns = scale_to_integers(patterns, processedinput)
patterns = np.ma.masked_array(patterns, mask=~mask)
print(patterns)
print(processedinput)
# Train NN2
nn2_model = train_nn2(patterns, processedinput, input_shape_nn2, output_shape_nn2)

# Predict text using NN2
predicted_text = predict_text(nn2_model, patterns)

print(reverse_preprocess(scale_to_integers(predicted_text, patterns), tokenizer))
"""
#scaled_data = scale_to_integers(predicted_text, processedinput)
scaled_data = predicted_text
nn3_model = train_nn3(extract_patterns(nn1_model, predicted_text), extract_patterns(nn1_model, processedinput), input_shape_nn3, output_shape_nn3)

# Evaluate correctness using NN3, correctness should be the output of the first neural network for the predicted and the expected
correctness = evaluate_correctness(nn3_model, extract_patterns(nn1_model, predicted_text), extract_patterns(nn1_model, processedinput))
print(correctness)
print(reverse_preprocess(scaled_data, tokenizer))
"""
