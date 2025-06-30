#!/usr/bin/env python3
"""
Script simple de entrenamiento para verificar que funciona
"""

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def train_simple_model():
    print("🧠 Entrenamiento simple para verificar funcionamiento...")
    
    # Parámetros
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 5  # Solo 5 epochs para probar rápido
    
    # Crear generadores
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Cargar datos
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'dataset/validation',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    print(f"✅ Train: {train_generator.samples} imágenes")
    print(f"✅ Validation: {validation_generator.samples} imágenes")
    print(f"✅ Clases: {len(train_generator.class_indices)}")
    
    # Modelo simple
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(train_generator.class_indices), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("🚀 Iniciando entrenamiento...")
    
    # Entrenar
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )
    
    # Guardar
    os.makedirs("models/saved_models", exist_ok=True)
    model_path = "models/saved_models/cartoon_classifier_simple.h5"
    model.save(model_path)
    
    print(f"✅ Modelo guardado: {model_path}")
    return model_path

if __name__ == "__main__":
    train_simple_model()