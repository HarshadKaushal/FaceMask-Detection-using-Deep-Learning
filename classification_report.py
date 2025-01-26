import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model('mask_detector_model.h5')

# Prepare the validation data
validation_dir = 'datasets/validation'  # Change this to your validation dataset path
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get true labels
y_true = validation_generator.classes  # True labels from the generator
y_pred = model.predict(validation_generator)  # Predictions from the model
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

# Generate and print the classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Optional: Display the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()