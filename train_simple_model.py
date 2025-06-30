#!/usr/bin/env python3
"""
Entrenador simple y rápido para el dataset de personajes
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json
from datetime import datetime

def create_simple_model(num_classes, img_size=(224, 224)):
    """Crear un modelo CNN simple y rápido"""
    
    model = models.Sequential([
        # Capas de entrada
        layers.Rescaling(1./255, input_shape=(*img_size, 3)),
        
        # Primera convolución
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda convolución
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Tercera convolución
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Clasificador
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def setup_data_simple(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=4):
    """Configurar datos de manera simple"""
    
    # Generadores simples
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen

def train_simple_model():
    """Función principal de entrenamiento simple"""
    
    print("🚀 ENTRENAMIENTO SIMPLE Y RÁPIDO")
    print("=" * 40)
    
    # Configuraciones
    img_size = (128, 128)  # Más pequeño = más rápido
    batch_size = 4         # Batch pequeño para poca memoria
    epochs = 20            # Menos épocas
    
    # Verificar directorios
    train_dir = "dataset/train"
    val_dir = "dataset/validation"
    test_dir = "dataset/test"
    
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            print(f"❌ Directorio no encontrado: {directory}")
            return
    
    print("📊 Configurando datos...")
    
    try:
        # Configurar datos
        train_gen, val_gen, test_gen = setup_data_simple(
            train_dir, val_dir, test_dir, img_size, batch_size
        )
        
        num_classes = len(train_gen.class_indices)
        class_names = list(train_gen.class_indices.keys())
        
        print(f"✅ Clases: {class_names}")
        print(f"📈 Train: {train_gen.samples} imágenes")
        print(f"📊 Val: {val_gen.samples} imágenes")
        print(f"🧪 Test: {test_gen.samples} imágenes")
        
        # Crear modelo
        print(f"🏗️ Creando modelo simple...")
        model = create_simple_model(num_classes, img_size)
        
        print(f"📋 Resumen del modelo:")
        model.summary()
        
        # Entrenar
        print(f"🚀 Entrenando por {epochs} épocas...")
        
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=1
        )
        
        print("✅ Entrenamiento completado!")
        
        # Evaluar
        print("📊 Evaluando modelo...")
        test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
        
        print(f"📈 Precisión en test: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        
        # Guardar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorios si no existen
        os.makedirs("models/saved_models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        model_path = f"models/saved_models/simple_cartoon_classifier_{timestamp}.h5"
        model.save(model_path)
        
        # Guardar metadata
        metadata = {
            'model_path': model_path,
            'class_names': class_names,
            'img_size': img_size,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'timestamp': timestamp,
            'model_type': 'simple_cnn'
        }
        
        metadata_path = "models/saved_models/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Guardar resultados
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'class_names': class_names,
            'model_type': 'simple_cnn'
        }
        
        results_path = "results/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 ¡ENTRENAMIENTO EXITOSO!")
        print(f"💾 Modelo guardado: {model_path}")
        print(f"📊 Precisión final: {test_accuracy:.4f}")
        print(f"📁 Metadata: {metadata_path}")
        print(f"📈 Resultados: {results_path}")
        
        # Predicción de ejemplo
        print(f"\n🧪 Probando predicción...")
        
        # Tomar una imagen de test
        test_gen.reset()
        x_test, y_test = next(test_gen)
        
        predictions = model.predict(x_test, verbose=0)
        predicted_class = np.argmax(predictions[0])
        actual_class = np.argmax(y_test[0])
        confidence = predictions[0][predicted_class]
        
        print(f"🎯 Predicción: {class_names[predicted_class]} (confianza: {confidence:.3f})")
        print(f"✅ Actual: {class_names[actual_class]}")
        
        if predicted_class == actual_class:
            print("🎉 ¡Predicción correcta!")
        else:
            print("❌ Predicción incorrecta")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        print(f"🔧 Intenta instalar dependencias: py -m pip install tensorflow")
        return False

if __name__ == "__main__":
    success = train_simple_model()
    
    if success:
        print(f"\n✅ Ejecuta 'py verify_training.py' para verificar los resultados")
    else:
        print(f"\n❌ El entrenamiento falló. Revisa los errores anteriores.")