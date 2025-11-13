import os
import shutil
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
project_root = r'D:\AI_Chinese_Handwrting_Recognition'  # Project root folder
dataset_root = r'D:\Traditional_Chinese_Data'  # Path to extracted dataset folder
num_classes = 100  # Set to 13065 for full; use 100 for testing
img_size = (64, 64)
batch_size = 32
epochs = 10
results_file = os.path.join(project_root, 'results.txt')  # Relative to project_root

# Ensure project folder exists
os.makedirs(project_root, exist_ok=True)
os.chdir(project_root)  # Switch to project dir for relative paths
print(f"Project set up in: {project_root}")

# Ensure GPU if available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def prepare_data():
    """Prepare train/test splits by copying first 40 images per class."""
    train_dir = os.path.join(project_root, 'train')
    test_dir = os.path.join(project_root, 'test')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get sorted subfolders (characters)
    subfolders = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])[:num_classes]
    print(f"Using {len(subfolders)} classes: {subfolders[:5]}...")  # Preview

    for class_idx, char_folder in enumerate(subfolders):
        char_path = os.path.join(dataset_root, char_folder)
        img_files = sorted(glob.glob(os.path.join(char_path, '*.png')))  # Sort alphabetically

        if len(img_files) < 40:
            print(f"Warning: {char_folder} has only {len(img_files)} images, skipping.")
            continue

        # Create class subdirs
        train_class_dir = os.path.join(train_dir, str(class_idx))
        test_class_dir = os.path.join(test_dir, str(class_idx))
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Copy first 40 to train
        for i in range(40):
            shutil.copy(img_files[i], os.path.join(train_class_dir, f'{char_folder}_{i}.png'))

        # Copy rest to test
        for i in range(40, len(img_files)):
            shutil.copy(img_files[i], os.path.join(test_class_dir, f'{char_folder}_{i}.png'))

    print(f"Data prepared: {num_classes} classes, train/test dirs created in {project_root}.")

def create_datagen():
    """Create ImageDataGenerators for train (augmented) and test."""
    train_dir = os.path.join(project_root, 'train')
    test_dir = os.path.join(project_root, 'test')

    # Train with augmentation (simulates ~200 samples/class via steps)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        shear_range=0.2,
        zoom_range=0.1,  # 0.9-1.1 scale
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        validation_split=0.0  # No extra split
    )

    # Test: only rescale
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Steps for ~200 samples/class (40 originals * 5 augmentations)
    steps_per_epoch = (40 * num_classes) // batch_size * 5  # Adjust multiplier for more augs

    return train_generator, test_generator, steps_per_epoch

def build_model1(input_shape, num_classes):  # Simple CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model2(input_shape, num_classes):  # Deeper with Dropout
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model3(input_shape, num_classes):  # With BatchNorm
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model_builder, name, train_gen, test_gen, steps_per_epoch):
    """Train model and return test accuracy."""
    model = model_builder((img_size[0], img_size[1], 3), num_classes)  # Note: generators yield RGB, but grayscale ok
    model.summary()

    # Callbacks - Save in project_root
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(os.path.join(project_root, f'{name}.h5'), monitor='val_accuracy', save_best_only=True)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_gen,
        callbacks=[early_stop, checkpoint]
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"{name} Test Accuracy: {test_acc:.4f}")

    # Plot history (save in project_root)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.savefig(os.path.join(project_root, f'{name}_history.png'))
    plt.close()

    return test_acc

if __name__ == "__main__":
    # Prepare data
    prepare_data()

    # Create generators
    train_gen, test_gen, steps = create_datagen()
    print(f"Train batches/epoch: {steps}, Test samples: {test_gen.samples}")

    # Train models
    models_acc = {}
    models_acc['Model1 (Simple)'] = train_and_evaluate(build_model1, 'model1', train_gen, test_gen, steps)
    models_acc['Model2 (Deeper Dropout)'] = train_and_evaluate(build_model2, 'model2', train_gen, test_gen, steps)
    models_acc['Model3 (BatchNorm)'] = train_and_evaluate(build_model3, 'model3', train_gen, test_gen, steps)

    # Find best
    best_model = max(models_acc, key=models_acc.get)
    best_acc = models_acc[best_model]
    shutil.copy(os.path.join(project_root, f'{best_model.lower().replace(" (", "").replace(")", "").replace(" ", "")}.h5'), os.path.join(project_root, 'best_model.h5'))

    # Log results
    with open(results_file, 'w') as f:
        f.write(f"Results for {num_classes} classes\n")
        f.write(f"Date: {datetime.now()}\n\n")
        for name, acc in models_acc.items():
            f.write(f"{name}: {acc:.4f}\n")
        f.write(f"\nBest: {best_model} with {best_acc:.4f}\n")

    print(f"\nResults logged to {results_file}. Best model: {best_model} ({best_acc:.4f})")
    print(f"Project complete! All files in: {project_root}")