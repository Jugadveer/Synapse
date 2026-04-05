import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 16

test_dir = "dataset/test"

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load model
model = tf.keras.models.load_model("dementia_model.h5")

# Evaluate
loss, acc = model.evaluate(test_data)

print(f"✅ Test Accuracy: {acc*100:.2f}%")