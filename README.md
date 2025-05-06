
# 🥔 Potato Disease Detection using CNN

This project focuses on detecting **potato leaf diseases** using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras. It classifies leaf images into one of the following categories:

- ✅ Healthy  
- 🌱 Early Blight  
- 🍂 Late Blight

---

## 📂 Dataset

The dataset should follow the structure:

```
potato_di/
├── Early Blight/
├── Late Blight/
└── Healthy/
```

You can load the dataset using TensorFlow's utility:

```python
tf.keras.preprocessing.image_dataset_from_directory(...)
```

The images are resized to **256x256**, and labels are automatically inferred from the folder names.

---

## 🧠 Model Architecture

A simple CNN is used, consisting of:

- 📦 Conv2D layers with ReLU activation  
- 🔽 MaxPooling2D for downsampling  
- 🧹 BatchNormalization and Dropout for regularization  
- 🔄 Flatten + Dense layers for classification  
- 🔚 Final Dense layer with 3 outputs (Softmax activation)

### 📌 Compilation:
- **Optimizer:** Adam  
- **Loss:** Sparse Categorical Crossentropy  
- **Metrics:** Accuracy

---

## 🏋️‍♀️ Training

- **Image Size:** 256x256  
- **Batch Size:** 32  
- **Epochs:** 50  
- Dataset is split into training, validation, and test sets.

```python
model.fit(train_dataset, validation_data=val_ds, epochs=50)
```

After training, the model is evaluated and tested on unseen images.

---

## 📈 Evaluation

The notebook includes:

- Training vs Validation Accuracy and Loss plots  
- Sample predictions with uploaded test images  
- Final evaluation on test dataset using `.evaluate()`

---

## 💾 Saving the Model

To use the trained model later, it is saved as:

```python
model.save("potato_disease_model.h5")
```

---

## 🚀 Web Application (Gradio)

You can run an interactive web app in Colab using Gradio:

```bash
pip install gradio
```

```python
import gradio as gr

def predict(image):
    # preprocess and predict
    return predicted_label

gr.Interface(fn=predict, inputs="image", outputs="text").launch()
```

You can also use `Streamlit` locally to build a GUI.

---

## 🔧 Requirements

Install all dependencies:

```bash
pip install tensorflow matplotlib gradio
```

---

## 🛠️ Future Enhancements

- Add data augmentation (flip, rotate, zoom)
- Use pre-trained models (MobileNet, EfficientNet)
- Deploy app using Streamlit, HuggingFace Spaces, or Flask
- Add performance metrics (Confusion Matrix, Precision, Recall)

---

## 📚 How to Use

1. Clone this repository:

```bash
git clone https://github.com/yourusername/potato-disease-detection.git
cd potato-disease-detection
```

2. Run the notebook in Google Colab or locally.

3. Upload your dataset (`potato_di.zip`) in the proper folder structure.

4. Train the model and evaluate performance.

---

## 👨‍💻 Author

- **Your Name**
- NIT Surat | Data Science Enthusiast

---

## 📃 License

This project is licensed under the MIT License.
