# Simple TensorFlow Model Creation

You're absolutely right! We can create a much simpler, more direct TensorFlow model. Here's a **clean, minimal approach** that cuts through all the complexity:

## 🎯 Simple Direct Approach

```python
# =========================
# SIMPLE DIRECT TENSORFLOW MODEL
# =========================

# 1. Simple imports
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from google.colab import files

print("TensorFlow version:", tf.__version__)

# 2. Upload data
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# 3. Load data
with open(file_name, 'r') as f:
    data = json.load(f)

questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

print(f"Loaded {len(questions)} Q&A pairs")

# 4. Simple tokenization
tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="")
tokenizer.fit_on_texts(questions + answers)

# Convert to sequences
X = tokenizer.texts_to_sequences(questions)
y = tokenizer.texts_to_sequences(answers)

# Pad sequences
MAX_LEN = 50
X = keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_LEN, padding='post')
y = keras.preprocessing.sequence.pad_sequences(y, maxlen=MAX_LEN, padding='post')

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Create simple model
vocab_size = len(tokenizer.word_index) + 1

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 128, input_length=MAX_LEN),
    keras.layers.LSTM(64),
    keras.layers.Dense(vocab_size * MAX_LEN),
    keras.layers.Reshape((MAX_LEN, vocab_size)),
    keras.layers.Softmax()
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Train
print("Training...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 8. Save everything
model.save("qa_model.h5")

# Save tokenizer
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

# Convert to TensorFlow.js
!pip install tensorflowjs
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "web_model")

# Package and download
!zip -r model_files.zip web_model tokenizer.json
files.download("model_files.zip")

print("✅ Done! Model ready for web deployment")
```


## 🚀 Super Simple Frontend


## ✅ Why This Direct Approach Works Better

1. **No GPU Configuration Issues** - Uses default settings
2. **No Complex Architecture** - Simple LSTM model
3. **No cuDNN Problems** - Avoids bidirectional layers
4. **Minimal Dependencies** - Only essential libraries
5. **Faster Training** - Smaller model, fewer epochs
6. **Easy Deployment** - Single zip file with everything

## 📋 Steps to Use

1. **Run the simple script** (either version)
2. **Upload your JSON file** 
3. **Wait 5-10 minutes** for training
4. **Download the zip file**
5. **Extract and host** the files
6. **Open the HTML file** in a browser

This direct approach eliminates all the complexity and gives you a working Q&A system in minutes!
