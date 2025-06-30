#!/usr/bin/env python3
"""
Script para entrenar el modelo de clasificación de personajes de dibujos animados
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

class CartoonCharacterTrainer:
    def __init__(self, 
                 train_dir="dataset/train",
                 val_dir="dataset/validation", 
                 test_dir="dataset/test",
                 model_save_dir="models/saved_models",
                 checkpoint_dir="models/checkpoints"):
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.model_save_dir = model_save_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Parámetros del modelo
        self.img_size = (224, 224)
        self.batch_size = 8  # Pequeño porque tienes pocos datos
        self.epochs = 50
        self.learning_rate = 0.001
        
        # Crear directorios
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)
        
        # Variables del modelo
        self.model = None
        self.history = None
        self.class_names = None
        
    def setup_data_generators(self):
        """Configurar generadores de datos con data augmentation"""
        
        print("📊 Configurando generadores de datos...")
        
        # Data augmentation para training (aumentar variedad)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Solo rescaling para validation y test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Generar datos
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Guardar nombres de clases
        self.class_names = list(train_generator.class_indices.keys())
        
        print(f"✅ Clases encontradas: {self.class_names}")
        print(f"📈 Datos de entrenamiento: {train_generator.samples} imágenes")
        print(f"📊 Datos de validación: {val_generator.samples} imágenes")
        print(f"🧪 Datos de prueba: {test_generator.samples} imágenes")
        
        return train_generator, val_generator, test_generator
    
    def create_model(self, num_classes):
        """Crear modelo usando transfer learning con MobileNetV2"""
        
        print(f"🏗️  Creando modelo para {num_classes} clases...")
        
        # Modelo base pre-entrenado
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Congelar capas base inicialmente
        base_model.trainable = False
        
        # Agregar capas personalizadas
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print("✅ Modelo creado exitosamente")
        model.summary()
        
        self.model = model
        return model
    
    def setup_callbacks(self):
        """Configurar callbacks para el entrenamiento"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks_list = [
            # Guardar mejor modelo
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, f'best_model_{timestamp}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Reducir learning rate cuando no mejore
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Parar early si no mejora
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Log del entrenamiento
            callbacks.CSVLogger(
                filename=f'results/training_log_{timestamp}.csv'
            )
        ]
        
        return callbacks_list
    
    def train_model(self, train_gen, val_gen):
        """Entrenar el modelo"""
        
        print(f"🚀 Iniciando entrenamiento por {self.epochs} épocas...")
        
        # Configurar callbacks
        callbacks_list = self.setup_callbacks()
        
        # Entrenar modelo
        history = self.model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history
        
        print("✅ Entrenamiento completado!")
        return history
    
    def fine_tune_model(self, train_gen, val_gen):
        """Fine-tuning del modelo (entrenar capas superiores del modelo base)"""
        
        print("🎯 Iniciando fine-tuning...")
        
        # Descongelar las últimas capas del modelo base
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Congelar todas las capas excepto las últimas 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompilar con learning rate más bajo
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Entrenar por menos épocas
        fine_tune_epochs = 10
        
        callbacks_list = self.setup_callbacks()
        
        history_fine = self.model.fit(
            train_gen,
            epochs=fine_tune_epochs,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Combinar historiales
        if self.history:
            for key in history_fine.history.keys():
                self.history.history[key].extend(history_fine.history[key])
        
        print("✅ Fine-tuning completado!")
        return history_fine
    
    def evaluate_model(self, test_gen):
        """Evaluar el modelo en datos de prueba"""
        
        print("📊 Evaluando modelo en datos de prueba...")
        
        # Evaluación básica
        test_loss, test_accuracy, test_top3 = self.model.evaluate(test_gen, verbose=1)
        
        print(f"📈 Precisión en prueba: {test_accuracy:.4f}")
        print(f"📈 Top-3 Accuracy: {test_top3:.4f}")
        
        # Predicciones para matriz de confusión
        test_gen.reset()
        predictions = self.model.predict(test_gen, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_gen.classes
        
        # Reporte de clasificación
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print("\n📋 Reporte de Clasificación:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Matriz de confusión
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Guardar resultados
        results = {
            'test_accuracy': float(test_accuracy),
            'test_top3_accuracy': float(test_top3),
            'test_loss': float(test_loss),
            'classification_report': report,
            'class_names': self.class_names
        }
        
        with open('results/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results, cm
    
    def plot_training_history(self):
        """Crear gráficos del entrenamiento"""
        
        if not self.history:
            print("⚠️  No hay historial de entrenamiento para graficar")
            return
        
        print("📈 Creando gráficos de entrenamiento...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Top-3 Accuracy
        axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Train Top-3')
        axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Val Top-3')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        
        # Learning Rate (si existe)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('results/plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Gráficos guardados en results/plots/")
    
    def plot_confusion_matrix(self, cm):
        """Crear gráfico de matriz de confusión"""
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('results/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_final_model(self):
        """Guardar modelo final"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_save_dir, f'cartoon_classifier_{timestamp}.h5')
        
        self.model.save(model_path)
        
        # Guardar también metadata
        metadata = {
            'model_path': model_path,
            'class_names': self.class_names,
            'img_size': self.img_size,
            'timestamp': timestamp,
            'num_classes': len(self.class_names)
        }
        
        with open(os.path.join(self.model_save_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Modelo guardado en: {model_path}")
        return model_path

def main():
    """Función principal de entrenamiento"""
    
    print("🎬 ENTRENAMIENTO DE CLASIFICADOR DE PERSONAJES")
    print("=" * 50)
    
    # Verificar que existan los directorios
    required_dirs = ["dataset/train", "dataset/validation", "dataset/test"]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Directorio no encontrado: {dir_path}")
            print("Ejecuta primero el script de preprocesamiento")
            return
    
    # Crear trainer
    trainer = CartoonCharacterTrainer()
    
    # Configurar datos
    train_gen, val_gen, test_gen = trainer.setup_data_generators()
    
    if len(trainer.class_names) == 0:
        print("❌ No se encontraron clases en los datos")
        return
    
    # Crear modelo
    model = trainer.create_model(len(trainer.class_names))
    
    # Entrenar modelo
    history = trainer.train_model(train_gen, val_gen)
    
    # Fine-tuning
    history_fine = trainer.fine_tune_model(train_gen, val_gen)
    
    # Evaluar modelo
    results, cm = trainer.evaluate_model(test_gen)
    
    # Crear gráficos
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(cm)
    
    # Guardar modelo final
    model_path = trainer.save_final_model()
    
    print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
    print(f"📈 Precisión final: {results['test_accuracy']:.4f}")
    print(f"💾 Modelo guardado en: {model_path}")
    print(f"📊 Resultados en: results/")

if __name__ == "__main__":
    main()