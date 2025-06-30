# Celda 2: Modelo Avanzado para Colab
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image

print("🚀 COLAB ADVANCED CARTOON CLASSIFIER")
print("="*50)
print(f"🔥 GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"🧠 TensorFlow Version: {tf.__version__}")

class ColabAdvancedTrainer:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 16  # Más grande en Colab
        self.epochs = 30
        self.learning_rate = 0.001
        
        # Directorios
        self.train_dir = "dataset/train"
        self.val_dir = "dataset/validation"
        self.test_dir = "dataset/test"
        
        self.model = None
        self.history = None
        self.class_names = None
        
    def setup_advanced_data_generators(self):
        """Data augmentation avanzado para Colab"""
        
        print("📊 Configurando generadores avanzados...")
        
        # Data augmentation intensivo
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            zoom_range=0.3,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
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
            batch_size=1,  # Para predicciones individuales
            class_mode='categorical',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        
        print(f"✅ Clases: {self.class_names}")
        print(f"📈 Train: {train_generator.samples} imágenes")
        print(f"📊 Val: {val_generator.samples} imágenes") 
        print(f"🧪 Test: {test_generator.samples} imágenes")
        
        return train_generator, val_generator, test_generator
    
    def create_advanced_model(self, num_classes, model_type='mobilenet'):
        """Crear modelo con transfer learning avanzado"""
        
        print(f"🏗️ Creando modelo {model_type}...")
        
        # Elegir arquitectura base
        if model_type == 'mobilenet':
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif model_type == 'efficientnet':
            base_model = EfficientNetB0(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        
        # Congelar modelo base
        base_model.trainable = False
        
        # Capas personalizadas
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compilar
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print("✅ Modelo creado!")
        model.summary()
        
        self.model = model
        return model
    
    def setup_callbacks(self):
        """Callbacks avanzados para Colab"""
        
        callbacks_list = [
            # Checkpoint del mejor modelo
            callbacks.ModelCheckpoint(
                filepath='best_cartoon_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Reducir LR dinámicamente
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate scheduler
            callbacks.LearningRateScheduler(
                lambda epoch: self.learning_rate * (0.9 ** epoch)
            )
        ]
        
        return callbacks_list
    
    def train_advanced_model(self, train_gen, val_gen):
        """Entrenamiento avanzado"""
        
        print(f"🚀 Entrenamiento avanzado - {self.epochs} épocas")
        
        callbacks_list = self.setup_callbacks()
        
        # Entrenar modelo congelado
        print("📚 Fase 1: Entrenamiento con base congelada")
        history1 = self.model.fit(
            train_gen,
            epochs=15,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Fine-tuning
        print("🎯 Fase 2: Fine-tuning")
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Congelar solo las primeras capas
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Recompilar con LR más bajo
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Continuar entrenamiento
        history2 = self.model.fit(
            train_gen,
            epochs=15,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Combinar historiales
        self.history = history1
        for key in history2.history.keys():
            self.history.history[key].extend(history2.history[key])
        
        return self.history
    
    def evaluate_and_visualize(self, test_gen):
        """Evaluación completa con visualizaciones"""
        
        print("📊 Evaluación completa...")
        
        # Cargar mejor modelo
        self.model = tf.keras.models.load_model('best_cartoon_model.h5')
        
        # Evaluar
        test_loss, test_accuracy, test_top3 = self.model.evaluate(test_gen, verbose=1)
        
        print(f"🎯 Precisión Final: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print(f"🔝 Top-3 Accuracy: {test_top3:.4f}")
        
        # Predicciones para matriz de confusión
        test_gen.reset()
        predictions = self.model.predict(test_gen, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_gen.classes
        
        # Reporte detallado
        report = classification_report(
            true_classes, predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print("\n📋 Reporte de Clasificación:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Matriz de confusión
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Visualizaciones
        self.plot_advanced_results(cm, report)
        
        return test_accuracy, report
    
    def plot_advanced_results(self, cm, report):
        """Gráficos profesionales"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Training History - Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training History - Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=axes[0, 2])
        axes[0, 2].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Predicted')
        axes[0, 2].set_ylabel('Actual')
        
        # 4. Precision por clase
        precisions = [report[cls]['precision'] for cls in self.class_names]
        axes[1, 0].bar(self.class_names, precisions, color='skyblue')
        axes[1, 0].set_title('Precision por Clase', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Recall por clase
        recalls = [report[cls]['recall'] for cls in self.class_names]
        axes[1, 1].bar(self.class_names, recalls, color='lightcoral')
        axes[1, 1].set_title('Recall por Clase', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. F1-Score por clase
        f1_scores = [report[cls]['f1-score'] for cls in self.class_names]
        axes[1, 2].bar(self.class_names, f1_scores, color='lightgreen')
        axes[1, 2].set_title('F1-Score por Clase', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('F1-Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_prediction_interface(self):
        """Interfaz para probar predicciones"""
        
        def predict_image(image_path):
            """Predecir una imagen"""
            
            # Cargar y procesar imagen
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predicción
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            # Mostrar resultado
            plt.figure(figsize=(10, 6))
            
            # Imagen
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f'Imagen a Clasificar', fontsize=14)
            plt.axis('off')
            
            # Predicciones
            plt.subplot(1, 2, 2)
            plt.bar(self.class_names, predictions[0])
            plt.title(f'Predicción: {predicted_class}\nConfianza: {confidence:.3f}', fontsize=14)
            plt.xticks(rotation=45)
            plt.ylabel('Probabilidad')
            
            plt.tight_layout()
            plt.show()
            
            return predicted_class, confidence
        
        return predict_image

# Función principal para ejecutar en Colab
def run_colab_training():
    """Ejecutar entrenamiento completo en Colab"""
    
    trainer = ColabAdvancedTrainer()
    
    # 1. Configurar datos
    train_gen, val_gen, test_gen = trainer.setup_advanced_data_generators()
    
    # 2. Crear modelo
    model = trainer.create_advanced_model(len(trainer.class_names), 'mobilenet')
    
    # 3. Entrenar
    history = trainer.train_advanced_model(train_gen, val_gen)
    
    # 4. Evaluar
    accuracy, report = trainer.evaluate_and_visualize(test_gen)
    
    # 5. Crear interfaz de predicción
    predict_fn = trainer.create_prediction_interface()
    
    print(f"\n🎉 ¡ENTRENAMIENTO COLAB COMPLETADO!")
    print(f"🎯 Precisión Final: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"💾 Modelo guardado: best_cartoon_model.h5")
    
    return trainer, predict_fn

# Ejecutar entrenamiento
trainer, predict_function = run_colab_training()