#!/usr/bin/env python3
"""
Entrenador mejorado de clasificador de personajes de dibujos animados
Optimizado para 20 imágenes por categoría
Autor: Tu proyecto cartoon-character-recognition
Fecha: Julio 2025
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import json
import shutil
from glob import glob

# Configurar paths del proyecto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
SCRIPTS_PATH = os.path.join(PROJECT_ROOT, 'scripts')

class CartoonClassifier:
    def __init__(self, img_size=(224, 224), num_classes=4):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = ['bob_esponja', 'dragon_ball_goku', 'mickey_mouse', 'pikachu']
        
        # Crear directorios necesarios
        os.makedirs(MODELS_PATH, exist_ok=True)
        os.makedirs(RESULTS_PATH, exist_ok=True)
        os.makedirs(os.path.join(MODELS_PATH, 'checkpoints'), exist_ok=True)
        
    def verify_dataset_structure(self):
        """Verifica la estructura del dataset"""
        print("🔍 VERIFICANDO ESTRUCTURA DEL DATASET")
        print("="*60)
        
        dataset_info = {}
        total_images = 0
        
        print(f"📁 Buscando en: {DATASET_PATH}")
        
        for class_name in self.class_names:
            class_path = os.path.join(DATASET_PATH, class_name)
            
            if os.path.exists(class_path):
                # Buscar imágenes con diferentes extensiones
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(glob(os.path.join(class_path, ext)))
                
                n_images = len(image_files)
                total_images += n_images
                dataset_info[class_name] = {
                    'count': n_images,
                    'path': class_path,
                    'files': image_files
                }
                
                status = "✅" if n_images >= 15 else "⚠️ " if n_images >= 10 else "❌"
                print(f"{status} {class_name}: {n_images} imágenes en {class_path}")
                
                if n_images < 10:
                    print(f"   💡 Recomendación: Agregar más imágenes a {class_name}")
                    
            else:
                dataset_info[class_name] = {'count': 0, 'path': class_path, 'files': []}
                print(f"❌ {class_name}: Carpeta no encontrada - {class_path}")
        
        print(f"\n📊 TOTAL: {total_images} imágenes")
        
        if total_images < 40:
            print("⚠️  Dataset pequeño. Se recomienda data augmentation intensivo.")
        elif total_images >= 80:
            print("✅ Dataset de buen tamaño para entrenamiento.")
        
        return dataset_info, total_images
    
    def organize_dataset_splits(self, dataset_info):
        """Organiza el dataset en train/validation/test manteniendo estructura original"""
        
        print("\n📁 ORGANIZANDO SPLITS DEL DATASET")
        print("-"*50)
        
        # Crear carpeta temporal para splits organizados
        splits_path = os.path.join(PROJECT_ROOT, 'dataset_splits')
        
        # Limpiar y crear estructura
        if os.path.exists(splits_path):
            shutil.rmtree(splits_path)
        
        for split in ['train', 'validation', 'test']:
            for class_name in self.class_names:
                os.makedirs(os.path.join(splits_path, split, class_name), exist_ok=True)
        
        split_summary = {}
        
        for class_name in self.class_names:
            images = dataset_info[class_name]['files']
            
            if len(images) == 0:
                continue
            
            # Mezclar imágenes de forma reproducible
            np.random.seed(42)
            shuffled_images = np.random.permutation(images).tolist()
            
            # División estratificada
            n_total = len(shuffled_images)
            n_train = max(1, int(0.7 * n_total))
            n_val = max(1, int(0.2 * n_total))
            
            train_images = shuffled_images[:n_train]
            val_images = shuffled_images[n_train:n_train+n_val]
            test_images = shuffled_images[n_train+n_val:]
            
            # Copiar archivos a splits
            for img_path in train_images:
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(splits_path, 'train', class_name, filename))
            
            for img_path in val_images:
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(splits_path, 'validation', class_name, filename))
            
            for img_path in test_images:
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(splits_path, 'test', class_name, filename))
            
            split_summary[class_name] = {
                'train': len(train_images),
                'validation': len(val_images),
                'test': len(test_images),
                'total': n_total
            }
            
            print(f"✅ {class_name}: {len(train_images)} train | {len(val_images)} val | {len(test_images)} test")
        
        return splits_path, split_summary
    
    def create_data_generators(self, splits_path):
        """Crea generadores de datos optimizados"""
        
        print("\n🔄 CREANDO GENERADORES DE DATOS")
        print("-"*40)
        
        # Data augmentation intensivo para dataset pequeño
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Solo normalización para validación y test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Crear generadores
        train_generator = train_datagen.flow_from_directory(
            os.path.join(splits_path, 'train'),
            target_size=self.img_size,
            batch_size=16,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        validation_generator = val_test_datagen.flow_from_directory(
            os.path.join(splits_path, 'validation'),
            target_size=self.img_size,
            batch_size=8,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(splits_path, 'test'),
            target_size=self.img_size,
            batch_size=1,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        print(f"🏋️  Entrenamiento: {train_generator.samples} imágenes")
        print(f"🔍 Validación: {validation_generator.samples} imágenes")
        print(f"🧪 Test: {test_generator.samples} imágenes")
        
        return train_generator, validation_generator, test_generator
    
    def build_improved_model(self):
        """Construye modelo mejorado con EfficientNet"""
        
        print("\n🤖 CONSTRUYENDO MODELO MEJORADO")
        print("-"*40)
        
        # Modelo base preentrenado
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Congelar base model inicialmente
        base_model.trainable = False
        
        # Arquitectura del modelo
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu', name='dense_256'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', name='dense_128'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ], name='cartoon_classifier')
        
        # Compilación inicial
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modelo construido con EfficientNetB0")
        print(f"📊 Parámetros totales: {self.model.count_params():,}")
        
        return self.model
    
    def setup_callbacks(self):
        """Configura callbacks para entrenamiento"""
        
        checkpoint_path = os.path.join(MODELS_PATH, 'checkpoints')
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_lr=1e-7,
                verbose=1,
                cooldown=3
            ),
            ModelCheckpoint(
                os.path.join(checkpoint_path, 'best_model_{epoch:02d}_{val_accuracy:.3f}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        return callbacks
    
    def train_model(self, train_gen, val_gen, epochs=120):
        """Entrena el modelo con estrategia de múltiples fases"""
        
        print("\n🚀 INICIANDO ENTRENAMIENTO MULTIFASE")
        print("="*60)
        
        # Configurar callbacks
        callbacks = self.setup_callbacks()
        
        # FASE 1: Entrenamiento con base congelada
        print("\n🔒 FASE 1: Transfer Learning (base congelada)")
        print("-" * 50)
        print("• Base model: CONGELADA")
        print("• Learning rate: 0.001")
        print("• Épocas: 40")
        
        history1 = self.model.fit(
            train_gen,
            epochs=40,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # FASE 2: Fine-tuning parcial
        print("\n🔓 FASE 2: Fine-tuning parcial")
        print("-" * 50)
        
        # Descongelar últimas capas del base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Congelar las primeras capas
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        print(f"• Capas entrenables: {sum(1 for layer in base_model.layers if layer.trainable)}/{len(base_model.layers)}")
        
        # Recompilar con learning rate menor
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("• Learning rate: 0.0001")
        print("• Épocas: 40 (total: 80)")
        
        history2 = self.model.fit(
            train_gen,
            epochs=40,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=40
        )
        
        # FASE 3: Fine-tuning completo (opcional)
        remaining_epochs = epochs - 80
        if remaining_epochs > 0:
            print("\n🔥 FASE 3: Fine-tuning completo")
            print("-" * 50)
            
            # Descongelar todas las capas
            for layer in base_model.layers:
                layer.trainable = True
            
            # Recompilar con learning rate muy bajo
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=0.00001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("• Todas las capas: ENTRENABLES")
            print("• Learning rate: 0.00001")
            print(f"• Épocas: {remaining_epochs} (total: {epochs})")
            
            history3 = self.model.fit(
                train_gen,
                epochs=remaining_epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=80
            )
            
            # Combinar historiales
            self.history = self.combine_histories([history1, history2, history3])
        else:
            # Solo dos fases
            self.history = self.combine_histories([history1, history2])
        
        print("\n✅ ENTRENAMIENTO COMPLETADO!")
        return self.history
    
    def combine_histories(self, histories):
        """Combina múltiples historiales de entrenamiento"""
        combined = {}
        
        for key in histories[0].history.keys():
            combined[key] = []
            for hist in histories:
                combined[key].extend(hist.history[key])
        
        # Crear objeto con estructura similar a history
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined)
    
    def plot_training_history(self):
        """Genera gráficos del entrenamiento"""
        
        if self.history is None:
            print("❌ No hay historial de entrenamiento disponible")
            return
        
        print("\n📊 GENERANDO GRÁFICOS DE ENTRENAMIENTO")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        
        # Accuracy
        ax1.plot(epochs, self.history.history['accuracy'], 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, self.history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(epochs, self.history.history['loss'], 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, self.history.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('Model Loss', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Métricas finales
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        
        ax3.bar(['Train Acc', 'Val Acc'], [final_train_acc, final_val_acc], 
                color=['blue', 'red'], alpha=0.7)
        ax3.set_ylim(0, 1)
        ax3.set_title('Final Accuracies', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Accuracy')
        
        # Resumen
        max_val_acc = max(self.history.history['val_accuracy'])
        min_val_loss = min(self.history.history['val_loss'])
        
        summary_text = f"""
        📊 RESUMEN DE ENTRENAMIENTO:
        
        • Max Val Accuracy: {max_val_acc:.4f}
        • Final Val Accuracy: {final_val_acc:.4f}
        • Min Val Loss: {min_val_loss:.4f}
        • Total Epochs: {len(epochs)}
        
        🎯 Diferencia Train-Val: {abs(final_train_acc - final_val_acc):.4f}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax4.set_title('Training Summary', fontweight='bold', fontsize=14)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_path = os.path.join(RESULTS_PATH, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {plot_path}")
        
        plt.show()
    
    def evaluate_model(self, test_gen):
        """Evalúa el modelo en el dataset de test"""
        
        print("\n🧪 EVALUACIÓN FINAL DEL MODELO")
        print("="*50)
        
        # Resetear generador y hacer predicciones
        test_gen.reset()
        predictions = self.model.predict(test_gen, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_gen.classes
        
        # Calcular accuracy
        test_accuracy = np.mean(predicted_classes == true_classes)
        print(f"\n🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Matriz de confusión
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Predicciones'})
        plt.title('Matriz de Confusión - Evaluación Final', fontweight='bold', fontsize=16)
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        
        # Guardar matriz de confusión
        cm_path = os.path.join(RESULTS_PATH, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz de confusión guardada: {cm_path}")
        
        plt.show()
        
        # Reporte de clasificación
        report = classification_report(true_classes, predicted_classes,
                                     target_names=self.class_names, output_dict=True)
        
        print("\n📋 REPORTE DE CLASIFICACIÓN:")
        print("-" * 60)
        
        # Mostrar reporte formateado
        print(f"{'Clase':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
        print("-" * 60)
        
        for class_name in self.class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                print(f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<8.0f}")
        
        # Métricas globales
        accuracy = report['accuracy']
        macro_avg = report['macro avg']
        
        print("-" * 60)
        print(f"{'Macro Avg':<15} {macro_avg['precision']:<10.3f} {macro_avg['recall']:<10.3f} {macro_avg['f1-score']:<10.3f}")
        print(f"{'Accuracy':<15} {accuracy:<10.3f}")
        
        # Guardar reporte
        report_path = os.path.join(RESULTS_PATH, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Reporte completo guardado: {report_path}")
        
        return predictions, predicted_classes, true_classes, report
    
    def save_model(self):
        """Guarda el modelo entrenado"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"cartoon_classifier_improved_{timestamp}.h5"
        model_path = os.path.join(MODELS_PATH, model_filename)
        
        # Guardar modelo
        self.model.save(model_path)
        
        # Guardar información del modelo
        model_info = {
            'model_file': model_filename,
            'classes': self.class_names,
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'training_date': datetime.now().isoformat(),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]) if self.history else None,
            'total_epochs': len(self.history.history['accuracy']) if self.history else None,
            'architecture': 'EfficientNetB0 + Custom Head'
        }
        
        info_path = os.path.join(MODELS_PATH, f"model_info_{timestamp}.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n💾 MODELO GUARDADO EXITOSAMENTE")
        print(f"📁 Archivo: {model_path}")
        print(f"📋 Info: {info_path}")
        
        return model_path, info_path

def main():
    """Función principal del entrenamiento"""
    
    print("🎭 CARTOON CHARACTER CLASSIFIER - VERSIÓN MEJORADA 🎭")
    print("="*70)
    print("📍 Directorio del proyecto:", PROJECT_ROOT)
    print("📁 Dataset esperado en:", DATASET_PATH)
    print()
    
    # Verificar TensorFlow y GPU
    print("🔧 CONFIGURACIÓN DEL SISTEMA:")
    print(f"• TensorFlow: {tf.__version__}")
    print(f"• GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU detectada - entrenamiento acelerado")
    else:
        print("⚠️  Solo CPU disponible - entrenamiento más lento")
    print()
    
    # Inicializar clasificador
    classifier = CartoonClassifier()
    
    try:
        # 1. Verificar dataset
        dataset_info, total_images = classifier.verify_dataset_structure()
        
        if total_images < 16:  # Mínimo 4 imágenes por clase
            print("\n❌ ERROR: Dataset insuficiente")
            print("💡 Necesitas al menos 4 imágenes por categoría")
            print("🎯 Recomendado: 20 imágenes por categoría")
            return
        
        # 2. Organizar splits
        splits_path, split_summary = classifier.organize_dataset_splits(dataset_info)
        
        # 3. Crear generadores
        train_gen, val_gen, test_gen = classifier.create_data_generators(splits_path)
        
        # 4. Construir modelo
        model = classifier.build_improved_model()
        
        # 5. Entrenar modelo
        epochs = 120  # Ajustable según necesidades
        print(f"\n⏱️  Iniciando entrenamiento por {epochs} épocas...")
        
        history = classifier.train_model(train_gen, val_gen, epochs=epochs)
        
        # 6. Visualizar resultados
        classifier.plot_training_history()
        
        # 7. Evaluación final
        predictions, pred_classes, true_classes, report = classifier.evaluate_model(test_gen)
        
        # 8. Guardar modelo
        model_path, info_path = classifier.save_model()
        
        # 9. Guardar historial completo
        history_path = os.path.join(RESULTS_PATH, f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(history_path, 'w') as f:
            json.dump({
                'history': classifier.history.history,
                'split_summary': split_summary,
                'dataset_info': {k: v['count'] for k, v in dataset_info.items()},
                'final_metrics': {
                    'val_accuracy': float(classifier.history.history['val_accuracy'][-1]),
                    'test_accuracy': float(np.mean(pred_classes == true_classes))
                }
            }, f, indent=2)
        
        # Resumen final
        final_val_acc = classifier.history.history['val_accuracy'][-1]
        test_acc = np.mean(pred_classes == true_classes)
        
        print("\n" + "="*70)
        print("🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("="*70)
        print(f"📊 Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"🧪 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"💾 Modelo guardado: {os.path.basename(model_path)}")
        print(f"📁 Resultados en: {RESULTS_PATH}")
        print()
        print("🚀 ¡Tu clasificador está listo para usar!")
        print("💡 Próximo paso: Subir a Google Colab para más experimentos")
        
        # Limpiar directorio temporal
        if os.path.exists(splits_path):
            shutil.rmtree(splits_path)
            print("🗑️  Limpieza: Directorio temporal removido")
        
    except Exception as e:
        print(f"\n❌ ERROR DURANTE EL ENTRENAMIENTO:")
        print(f"📋 {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Limpiar en caso de error
        splits_path = os.path.join(PROJECT_ROOT, 'dataset_splits')
        if os.path.exists(splits_path):
            shutil.rmtree(splits_path)

if __name__ == "__main__":
    main()