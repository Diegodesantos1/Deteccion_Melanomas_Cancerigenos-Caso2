import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Evaluar el modelo en datos de prueba
test_loss, test_acc = model.evaluate(test_generator)
print(f'Precisión en el conjunto de prueba: {test_acc}')
