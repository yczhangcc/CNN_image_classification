# 🌟 CNN Image Classification 🌟

Welcome to the **CNN Image Classification** project! This repository showcases a Convolutional Neural Network (CNN) model built using TensorFlow to classify images into different categories, such as 🌻 sunflowers, 🌹 roses, and 🦋 dandelions.

## 🚀 Project Overview

- **Task**: Train a CNN model to classify images into different categories.
- **Dataset**: Custom dataset crawled from the web (e.g., Baidu Images) with at least three categories and 400+ images per category.
- **Model**: A CNN architecture built using TensorFlow and Keras for image classification.

### 🛠️ Steps:

1. **🔍 Data Collection**: Crawled images from Baidu Images for multiple categories.
2. **🧼 Data Preprocessing**: Split images into training, validation, and testing sets.
3. **🏗️ Model Architecture**: Designed a CNN model with convolutional, pooling, and fully connected layers.
4. **🧑‍💻 Training**: Trained the model on the dataset and saved it for future use.
5. **📈 Evaluation**: Evaluated the model accuracy on the test set.

---

## 📂 Folder Structure

- 📁 `SplitData/`: Contains the split data for training, validation, and testing.
- 📝 `cnn_test.py`: Script to test the model accuracy using the test dataset.
- 📝 `cnn_train.py`: Script to train the CNN model.
- 📝 `data_split.py`: Script to split the data into training, validation, and testing sets.
- 📝 `get_data.py`: Script to retrieve or preprocess the dataset.
- 📊 `results_cnn.png`: A plot visualizing the training and testing results.

---

## 🛠️ Setup and Installation

### Prerequisites:

- 🐍 **Python** (version 3.8)
- 🧠 **TensorFlow** (version 2.x)
- Other dependencies such as `matplotlib`, `scikit-learn`, `opencv-python`

### Installation:

1. **Clone this repository**:

   ```bash
   git clone https://github.com/yourusername/CNN_image_classification.git
   cd CNN_image_classification
   ```

2. **Set up a Python virtual environment**:

   ```bash
   conda create -n py38 python=3.8
   conda activate py38
   ```

3. **Install the required dependencies**:

   ```bash
   pip install tensorflow-cpu scikit-learn matplotlib seaborn pandas openpyxl opencv-python
   ```

4. **(Optional) Install PyTorch for additional analysis**:

   ```bash
   pip install torch torchvision torchaudio
   ```

---

## 💻 Usage

1. **📂 Data Splitting**:

   Run the `data_split.py` script to split your dataset into training, validation, and testing sets:

   ```bash
   python data_split.py
   ```

2. **🏋️‍♀️ Model Training**:

   Run the `cnn_train.py` script to start the model training:

   ```bash
   python cnn_train.py
   ```

3. **🧪 Model Testing**:

   After training, test the model using `cnn_test.py`:

   ```bash
   python cnn_test.py
   ```

---

## 📊 Results

The results from the training process are saved and visualized in `results_cnn.png`. You can evaluate model performance and accuracy through this visual representation.

---

## 🌐 References

- [🔗 Baidu Images for Crawling](https://baidu.com)
- [📚 TensorFlow Documentation](https://tensorflow.org)

---

## 🌸 Icons and Indicators

- 🌻 **Sunflowers**: One of the image categories.
- 🌹 **Roses**: Another image category.
- 🦋 **Dandelions**: A third image category.
- 📊 **Model Results**: A visual indicator of training performance.
- 🧑‍💻 **Model Training**: Steps for building and training the model.
- 🧪 **Model Testing**: Steps for evaluating model performance.

---

Feel free to explore the repository and enjoy the journey of building your own image classifier! 🚀

---

I hope this version adds more visual fun and interest. Let me know if you'd like further tweaks! 😊
