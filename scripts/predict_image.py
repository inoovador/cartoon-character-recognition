#!/usr/bin/env python3
"""
Script de predicción para el clasificador de personajes de dibujos animados
Uso: python scripts/predict_image.py [ruta_imagen] [--model modelo.h5]
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Agregar el directorio raíz al path para importaciones
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def load_latest_model():
    """Carga el modelo más reciente"""
    
    models_path = os.path.join(PROJECT_ROOT, 'models')
    
    # Buscar modelos .h5
    model_files = [f for f in os.listdir(models_path) if f.endswith('.h5') and 'cartoon_classifier' in f]
    
    if not model_files:
        raise FileNotFoundError("No se encontraron modelos entrenados en la carpeta models/")
    
    # Ordenar por fecha de modificación (más reciente primero)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_path, x)), reverse=True)
    latest_model = model_files[0]
    
    model_path = os.path.join(models_path, latest_model)
    
    print(f"📍 Cargando modelo: {latest_model}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo cargado exitosamente")
        return model, latest_model
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {str(e)}")

def predict_image(model, image_path, img_size=(224, 224)):
    """Predice la clase de una imagen"""
    
    class_names = ['bob_esponja', 'dragon_ball_goku', 'mickey_mouse', 'pikachu']
    
    # Verificar que existe la imagen
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    
    try:
        # Cargar y preprocesar imagen
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Hacer predicción
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = class_names[predicted_class_idx]
        
        # Crear diccionario de probabilidades
        probabilities = {name: float(prob) for name, prob in zip(class_names, predictions[0])}
        
        result = {
            'imagen': os.path.basename(image_path),
            'clase_predicha': predicted_class,
            'confianza': float(confidence),
            'probabilidades': probabilities,
            'timestamp': datetime.now().isoformat()
        }
        
        return result, img
        
    except Exception as e:
        raise Exception(f"Error al procesar la imagen: {str(e)}")

def visualize_prediction(result, img, save_plot=False):
    """Visualiza la predicción"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mostrar imagen
    ax1.imshow(img)
    ax1.set_title(f'Imagen: {result["imagen"]}', fontweight='bold')
    ax1.axis('off')
    
    # Mostrar predicción con estilo
    prediction_text = f'{result["clase_predicha"].replace("_", " ").title()}\nConfianza: {result["confianza"]:.1%}'
    
    # Color basado en confianza
    if result["confianza"] > 0.8:
        color = 'green'
        emoji = '🎯'
    elif result["confianza"] > 0.6:
        color = 'orange'
        emoji = '🤔'
    else:
        color = 'red'
        emoji = '❓'
    
    ax1.text(0.02, 0.98, f'{emoji} {prediction_text}', 
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', color=color,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Gráfico de barras con probabilidades
    classes = list(result["probabilidades"].keys())
    probs = list(result["probabilidades"].values())
    
    # Colores para cada clase
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars = ax2.bar(classes, probs, color=colors, alpha=0.7)
    ax2.set_title('Probabilidades por Clase', fontweight='bold')
    ax2.set_ylabel('Probabilidad')
    ax2.set_ylim(0, 1)
    
    # Rotar etiquetas del eje x
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Agregar valores en las barras
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Resaltar la clase predicha
    predicted_idx = classes.index(result["clase_predicha"])
    bars[predicted_idx].set_alpha(1.0)
    bars[predicted_idx].set_edgecolor('black')
    bars[predicted_idx].set_linewidth(2)
    
    plt.tight_layout()
    
    if save_plot:
        results_path = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(results_path, exist_ok=True)
        
        plot_filename = f'prediction_{result["imagen"].split(".")[0]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plot_path = os.path.join(results_path, plot_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado: {plot_path}")
    
    plt.show()

def save_prediction_log(result):
    """Guarda el log de predicciones"""
    
    results_path = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_path, exist_ok=True)
    
    log_file = os.path.join(results_path, 'prediction_log.json')
    
    # Cargar log existente o crear nuevo
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {'predictions': []}
    
    # Agregar nueva predicción
    log_data['predictions'].append(result)
    
    # Guardar log actualizado
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"📝 Predicción guardada en log: {log_file}")

def main():
    """Función principal"""
    
    parser = argparse.ArgumentParser(description='Predice la clase de una imagen usando el clasificador entrenado')
    parser.add_argument('image_path', help='Ruta a la imagen a clasificar')
    parser.add_argument('--model', help='Ruta específica al modelo (opcional)')
    parser.add_argument('--save-plot', action='store_true', help='Guardar gráfico de resultado')
    parser.add_argument('--save-log', action='store_true', help='Guardar en log de predicciones')
    parser.add_argument('--no-display', action='store_true', help='No mostrar gráficos (para uso en servidor)')
    
    args = parser.parse_args()
    
    print("🎭 PREDICTOR DE PERSONAJES DE DIBUJOS ANIMADOS")
    print("="*60)
    
    try:
        # Cargar modelo
        if args.model:
            if not os.path.exists(args.model):
                raise FileNotFoundError(f"Modelo no encontrado: {args.model}")
            model = tf.keras.models.load_model(args.model)
            model_name = os.path.basename(args.model)
            print(f"📍 Modelo especificado cargado: {model_name}")
        else:
            model, model_name = load_latest_model()
        
        # Hacer predicción
        print(f"🔍 Analizando imagen: {args.image_path}")
        result, img = predict_image(model, args.image_path)
        
        # Mostrar resultado
        print("\n" + "="*50)
        print("🎯 RESULTADO DE LA PREDICCIÓN")
        print("="*50)
        print(f"📁 Imagen: {result['imagen']}")
        print(f"🏷️  Clase predicha: {result['clase_predicha'].replace('_', ' ').title()}")
        print(f"🎲 Confianza: {result['confianza']:.1%}")
        print("\n📊 Probabilidades detalladas:")
        
        for clase, prob in sorted(result['probabilidades'].items(), key=lambda x: x[1], reverse=True):
            emoji = "🥇" if clase == result['clase_predicha'] else "  "
            print(f"  {emoji} {clase.replace('_', ' ').title()}: {prob:.1%}")
        
        # Visualizar resultado
        if not args.no_display:
            visualize_prediction(result, img, save_plot=args.save_plot)
        
        # Guardar log si se especifica
        if args.save_log:
            save_prediction_log(result)
        
        print(f"\n✅ Predicción completada exitosamente!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()