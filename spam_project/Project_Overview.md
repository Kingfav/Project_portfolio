#  Spam Detection Web App

A simple and interactive web application that detects whether a given message is **Spam** or **Not Spam** using a machine learning model built with Python.

##  Project Overview

This project combines **machine learning** and **web development** to simulate a real-world spam filtering system. Users can enter any message, and the app instantly classifies it using a trained model with a confidence score.

The project demonstrates full-stack integration:
- Machine Learning (model training & prediction)
- Flask (backend API)
- HTML/CSS/JavaScript (frontend interface)

---

##  Features

-  Real-time spam detection
-  ML model trained on labeled text data
-  Confidence score display
-  Clean, responsive UI
-  Frontend-to-backend communication using `fetch()`

---

##  Technologies Used

- **Python** – model training, data processing
- **Scikit-learn** – vectorization and classification
- **Flask** – lightweight backend API
- **HTML/CSS** – frontend layout and design
- **JavaScript** – frontend logic & API communication
- **Pickle** – to serialize and load the model/vectorizer

---

##  How It Works

1. A user types or pastes a message into the input box.
2. JavaScript sends the message to the Flask server via a POST request to `/predict`.
3. The Flask backend loads the pre-trained model and vectorizer.
4. The message is transformed and classified as **Spam** or **Not Spam**.
5. The result (plus confidence score) is displayed on the webpage.

---

## 📁 Project Structure

