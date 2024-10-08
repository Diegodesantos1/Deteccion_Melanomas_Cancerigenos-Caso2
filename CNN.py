import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Definir paths de entrenamiento y prueba con subdirectorios 'Benign' y 'Malignant'
train_dir = 'train'  # Carpeta principal con subcarpetas 'Benign' y 'Malignant'
test_dir = 'test'    # Carpeta principal con subcarpetas 'Benign' y 'Malignant'

# Preprocesamiento de imágenes
train_datagen = ImageDataGenerator(rescale=1/255.0, rotation_range=40, 
                                   width_shift_range=0.2, height_shift_range=0.2, 
                                   shear_range=0.2, zoom_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1/255.0)

# Cargar imágenes en lotes desde las carpetas 'Benign' y 'Malignant'
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Clasificación binaria (0: Benign, 1: Malignant)
    classes=['Benign', 'Malignant']  # Especificar clases
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Clasificación binaria (0: Benign, 1: Malignant)
    classes=['Benign', 'Malignant']  # Especificar clases
)


# Construcción del modelo CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Salida binaria (benigno/maligno)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_generator, epochs=30, validation_data=test_generator)



# Obtener los datos del historial de entrenamiento
acc = history.history['accuracy']  # Precisión en entrenamiento
val_acc = history.history['val_accuracy']  # Precisión en validación

loss = history.history['loss']  # Pérdida en entrenamiento
val_loss = history.history['val_loss']  # Pérdida en validación

epochs_range = range(len(acc))

# Gráfica de Train Accuracy vs Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, acc, label='Train Accuracy', color='blue')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange')
plt.title('Train vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Gráfica de Train Loss vs Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, loss, label='Train Loss', color='blue')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange')
plt.title('Train vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()