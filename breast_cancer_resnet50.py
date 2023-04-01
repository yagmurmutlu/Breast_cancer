import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

# Define dataset path and batch size
train_data_dir = ''
test_data_dir = ''
batch_size = 32

# Define image size
img_height = 224
img_width = 224

# Create data generators for training, validation, and test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')



test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Load pre-trained ResNet50 model
resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers
for layer in resnet.layers:
    layer.trainable = False

# Add new output layer
x = resnet.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)

# Create model
model = tf.keras.models.Model(inputs=resnet.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
# Train model
model.fit(
    train_generator,
    epochs=epochs,
    
)

# Evaluate model on test set
y_true = []
y_pred = []
for i in range(len(test_generator)):
    x, y = test_generator.next()
    y_true.extend(np.argmax(y, axis=1))
    y_pred.extend(np.argmax(model.predict(x), axis=1))

f1 = f1_score(y_true, y_pred, average='weighted')
plt.plot([f1], 'ro', markersize=10)
plt.title('F1 Score')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.show()
print('F1 score:', f1)
