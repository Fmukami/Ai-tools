# Ai-tools
Primary differences between TensorFlow and PyTorch & when to choose

Programming style:

TensorFlow: Uses static computation graphs (TF1.x) and now also supports eager execution (TF2.x). More production-focused with deployment tools like TensorFlow Serving and TensorFlow Lite.

PyTorch: Uses dynamic computation graphs (define-by-run), making it more intuitive and Pythonic for experimentation and debugging.

Ecosystem:

TensorFlow: Strong integration with Google’s tools, TPU support, and mature production pipelines.

PyTorch: Favored in research, fast prototyping, and academic work; widely adopted for flexibility.

When to choose:

TensorFlow: When production deployment, scalability, and cross-platform support are priorities.

PyTorch: When doing rapid prototyping, research, or needing more intuitive debugging.

Q2: Two use cases for Jupyter Notebooks in AI development

Interactive prototyping and data exploration — Run code cells incrementally, visualize datasets, and adjust models without restarting the whole program.

Model training demonstrations and documentation — Combine code, text, and visual outputs in one place for tutorials, reports, and reproducible research.

Q3: How spaCy enhances NLP compared to basic Python string operations

spaCy provides advanced, efficient NLP pipelines (tokenization, part-of-speech tagging, named entity recognition, dependency parsing) that are language-aware and optimized for speed.

Basic Python string ops (like .split() or .replace()) treat text as raw sequences of characters without linguistic context, making them inadequate for complex NLP tasks.

In short: spaCy understands language structure, not just raw text patterns.

Comparative Analysis — Scikit-learn vs TensorFlow

Feature	Scikit-learn	TensorFlow
Target applications	Classical ML (linear regression, decision trees, clustering, etc.)	Deep learning (neural networks, CNNs, RNNs, transformers)
Ease of use	Very beginner-friendly; consistent APIs; minimal setup needed	Steeper learning curve; more complex model setup
Community support	Large, mature, well-documented; strong for traditional ML	Large and active, especially in DL/AI production environments


# Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
iris = load_iris(as_frame=True)
df = iris.frame.copy()

# 2. Simulate missing values for demonstration (optional)
np.random.seed(0)
df.iloc[::15, 0] = np.nan  # NaN in sepal length

# 3. Handle missing values
imputer = SimpleImputer(strategy='mean')
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# 4. Encode labels
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# 5. Split data
X = df.iloc[:, :-1]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 7. Predict and Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("Decision Tree Results:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (samples, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# 3. Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, batch_size=64)

# 5. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 6. Visualize predictions
def plot_samples(X, y_true, y_pred, sample_idxs):
    plt.figure(figsize=(10,2))
    for i, idx in enumerate(sample_idxs):
        plt.subplot(1, 5, i+1)
        plt.imshow(X[idx].reshape(28,28), cmap='gray')
        plt.title(f"Label: {y_true[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    plt.show()

sample_idxs = np.random.choice(len(x_test), 5, replace=False)
preds = np.argmax(model.predict(x_test[sample_idxs]), axis=1)
plot_samples(x_test, y_test, preds, sample_idxs)

import spacy
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher

# Install if missing: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Sample Amazon reviews
reviews = [
    "I love my new Apple iPhone! The camera is fantastic.",
    "The Samsung Galaxy earbuds are terrible. Not worth the price.",
    "Sony headphones deliver amazing sound quality."
]

# Custom product/brand matcher (expand as needed)
brands = ['Apple', 'Samsung', 'Sony', 'iPhone', 'Galaxy', 'headphones', 'earbuds']
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
patterns = [nlp.make_doc(text) for text in brands]
matcher.add("PRODUCT_BRAND", patterns)

def extract_entities(text):
    doc = nlp(text)
    matches = matcher(doc)
    entities = set()
    for match_id, start, end in matches:
        entities.add(doc[start:end].text)
    return list(entities)

# Simple rule-based sentiment analysis
def get_sentiment(text):
    positive = ['love', 'fantastic', 'amazing', 'great']
    negative = ['terrible', 'not worth', 'bad', 'poor']
    text_lower = text.lower()
    if any(word in text_lower for word in positive):
        return "positive"
    elif any(word in text_lower for word in negative):
        return "negative"
    return "neutral"

for review in reviews:
    entities = extract_entities(review)
    sentiment = get_sentiment(review)
    print(f"Review: {review}")
    print(f"Extracted Entities: {entities}")
    print(f"Sentiment: {sentiment}")
    print("----------")

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load trained model
model = tf.keras.models.load_model('mnist_model.h5')

st.title("MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale image of a digit.")

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image)  # Invert colors if needed
    image = image.resize((28,28))
    st.image(image, caption='Uploaded Digit', use_column_width=True)
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1,28,28,1)
    prediction = np.argmax(model.predict(img_array), axis=1)[0]
    st.write(f"Predicted Digit: {prediction}")
