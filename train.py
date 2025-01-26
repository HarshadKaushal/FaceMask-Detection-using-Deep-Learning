from Mask import MaskDetector  
import matplotlib.pyplot as plt


detector = MaskDetector()

# Create the model
model = detector.create_model()

# Print model summary to see the architecture
model.summary()

# Prepare the data
train_dir = 'datasets/train'
validation_dir = 'datasets/validation'
train_generator, validation_generator = detector.prepare_data(train_dir, validation_dir)

# Train the model
history = detector.train(train_generator, validation_generator, epochs=10)

# Plot training results
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('mask_detector_model.h5')     