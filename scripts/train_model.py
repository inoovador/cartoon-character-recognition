#!/usr/bin/env python3
"""
Script para entrenar modelo de reconocimiento de personajes
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Configurar GPU si está disponible
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"🚀 GPU disponible: {physical_devices[0]}")
else:
    print("💻 Entrenando en CPU")

class CharacterClassifier:
    def __init__(self, 
                 train_dir="dataset/train",
                 val_dir="dataset/validation", 
                 test_dir="dataset/test",
                 img_size=(224, 224),
                 batch_size=32):
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Crear directorios de salida
        os.makedirs("models/checkpoints", exist_ok=True)
        os.makedirs("models/saved_models", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)
        os.makedirs("results/reports", exist_ok=True)
        
        # Obtener número de clases
        self.character_names = sorted(os.listdir(train_dir))
        self.num_classes = len(self.character_names)
        
        print(f"📊 Detectados {self.num_classes} personajes:")
        for i, char in enumerate(self.character_names):
            count = len(os.listdir(os.path.join(train_dir, char)))
            print(f"   {i+1:2d}. {char}: {count} imágenes")
    
    def create_data_generators(self):
        """Crear generadores de datos con aumento"""
        
        # Generador para entrenamiento (con aumento de datos)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Generador para validación/test (solo normalización)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Crear generadores
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"🎯 Generadores creados:")
        print(f"   - Train: {self.train_generator.samples} imágenes")
        print(f"   - Validation: {self.val_generator.samples} imágenes") 
        print(f"   - Test: {self.test_generator.samples} imágenes")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def create_cnn_model(self, model_type="basic"):
        """Crear modelo CNN"""
        
        if model_type == "basic":
            model = models.Sequential([
                # Bloque 1
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Bloque 2
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Bloque 3
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Bloque 4
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling2D(),
                
                # Clasificador
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        elif model_type == "transfer":
            # Usar modelo preentrenado
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compilar modelo
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        self.model = model
        print(f"🧠 Modelo {model_type} creado:")
        print(f"   - Parámetros: {model.count_params():,}")
        
        return model
    
    def train_model(self, epochs=50, model_type="basic"):
        """Entrenar el modelo"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"cartoon_classifier_{model_type}_{timestamp}"
        
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=f"models/checkpoints/{model_name}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        print(f"🚀 Iniciando entrenamiento de {epochs} epochs...")
        
        # Entrenar
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Guardar modelo final
        final_model_path = f"models/saved_models/{model_name}_final.h5"
        self.model.save(final_model_path)
        
        # Guardar historial
        history_path = f"results/reports/{model_name}_history.json"
        with open(history_path, 'w') as f:
            # Convertir arrays numpy a listas para JSON
            hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
            json.dump(hist_dict, f, indent=2)
        
        self.history = history
        self.model_name = model_name
        
        print(f"✅ Entrenamiento completado!")
        print(f"📁 Modelo guardado: {final_model_path}")
        
        return history
    
    def plot_training_history(self):
        """Graficar historial de entrenamiento"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0,0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Loss
        axes[0,1].plot(self.history.history['loss'], label='Train Loss')
        axes[0,1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0,1].set_title('Model Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Top-5 Accuracy
        if 'top_5_accuracy' in self.history.history:
            axes[1,0].plot(self.history.history['top_5_accuracy'], label='Train Top-5')
            axes[1,0].plot(self.history.history['val_top_5_accuracy'], label='Val Top-5')
            axes[1,0].set_title('Top-5 Accuracy')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Accuracy')
            axes[1,0].legend()
            axes[1,0].grid(True)
        
        # Learning Rate (si está disponible)
        axes[1,1].axis('off')
        axes[1,1].text(0.5, 0.5, f'Final Training Accuracy: {self.history.history["accuracy"][-1]:.4f}\n'
                                 f'Final Validation Accuracy: {self.history.history["val_accuracy"][-1]:.4f}',
                      transform=axes[1,1].transAxes, ha='center', va='center',
                      fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plot_path = f"results/plots/{self.model_name}_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Gráficos guardados: {plot_path}")
    
    def evaluate_model(self):
        """Evaluar modelo en conjunto de test"""
        
        print("🧪 Evaluando modelo en conjunto de test...")
        
        # Predicciones
        test_loss, test_acc, test_top5 = self.model.evaluate(self.test_generator, verbose=1)
        
        # Obtener predicciones detalladas
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Reporte de clasificación
        class_report = classification_report(
            true_classes, predicted_classes, 
            target_names=self.character_names,
            output_dict=True
        )
        
        # Guardar reporte
        report_path = f"results/reports/{self.model_name}_classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(class_report, f, indent=2)
        
        # Matriz de confusión
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Visualizar matriz de confusión
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.character_names,
                   yticklabels=self.character_names)
        plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        confusion_path = f"results/plots/{self.model_name}_confusion_matrix.png"
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Resultados del Test:")
        print(f"   - Accuracy: {test_acc:.4f}")
        print(f"   - Top-5 Accuracy: {test_top5:.4f}")
        print(f"   - Loss: {test_loss:.4f}")
        print(f"📁 Reportes guardados en results/reports/")
        
        return test_acc, test_top5, class_report

def main():
    """Función principal de entrenamiento"""
    
    print("🎭 ENTRENAMIENTO DE CLASIFICADOR DE PERSONAJES")
    print("=" * 60)
    
    # Crear clasificador
    classifier = CharacterClassifier()
    
    # Crear generadores de datos
    train_gen, val_gen, test_gen = classifier.create_data_generators()
    
    # Elegir tipo de modelo
    model_type = input("\n🧠 Tipo de modelo (basic/transfer): ").strip().lower()
    if model_type not in ['basic', 'transfer']:
        model_type = 'basic'
    
    # Crear modelo
    model = classifier.create_cnn_model(model_type=model_type)
    
    # Mostrar arquitectura
    print(f"\n📋 Arquitectura del modelo:")
    model.summary()
    
    # Entrenar
    epochs = int(input(f"\n⏱️  Número de epochs (recomendado: 30-50): ") or 30)
    history = classifier.train_model(epochs=epochs, model_type=model_type)
    
    # Visualizar entrenamiento
    classifier.plot_training_history()
    
    # Evaluar
    test_acc, test_top5, report = classifier.evaluate_model()
    
    print(f"\n🎉 ¡Entrenamiento completado exitosamente!")
    print(f"🏆 Accuracy final en test: {test_acc:.4f}")
    print(f"🥇 Top-5 accuracy: {test_top5:.4f}")

if __name__ == "__main__":
    main()