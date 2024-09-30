# Brain-tumor-classification-using-CNN


## Dataset
The dataset used in this project consists of brain MRI images categorized into different types of tumors. The data includes both tumor-positive and tumor-negative images to train and evaluate the model effectively.

- **Source**: The dataset can be accessed at the following DOI: [10.34740/kaggle/dsv/2127923]
- **Categories**: Tumor types such as meningioma, glioma, pituitary tumor, etc.

## Goal
The objective of this project is to develop a deep learning model to classify brain MRI images and detect different types of brain tumors. Early detection of tumors can aid in better diagnosis and treatment.

## Methodology
1. **Data Preprocessing**: 
    - Normalized and resized the MRI images.
    - Applied data augmentation techniques to improve model generalization.
2. **Model Architecture**:
    - Built a Convolutional Neural Network (CNN) using the Keras library.
    - Used multiple convolutional layers followed by max-pooling and dropout layers.
    - Implemented dense fully connected layers for final classification.
3. **Training**:
    - Split data into training, validation, and test sets.
    - Used appropriate loss function and optimizer (e.g., Adam) to train the model.
    - Hyperparameter tuning for optimal performance.
4. **Evaluation**:
    - Evaluated the model using metrics like accuracy, precision, recall.
    - Cross-validated the model with different test sets.

## Dependencies
To run this project, ensure you have the following libraries installed:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn


## Results
- Achieved an accuracy of **90.74%** on the test set.
- The model successfully classified brain MRI images with significant accuracy for each tumor category.

| Metric       | Value   |
| ------------ | ------- |
| Accuracy     | 90.74%      |
| Precision    | 93.83%      |
| Recall       | 93.83%      |

## Future Work
- **Model Improvement**: Experiment with deeper architectures such as ResNet or VGGNet for better accuracy.
- **Hyperparameter Tuning**: Further optimize learning rates, batch sizes, and other hyperparameters.
- **Transfer Learning**: Explore the use of pre-trained models like VGG16 or EfficientNet for transfer learning.
- **Additional Features**: Explore incorporating other medical data (e.g., patient demographics) to enhance model performance.
