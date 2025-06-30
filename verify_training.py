#!/usr/bin/env python3
"""
Script para verificar que el entrenamiento se completó correctamente
"""

import os
import json
import glob
from datetime import datetime

def verify_training_completion():
    """Verificar que todos los archivos esperados existan y sean válidos"""
    
    print("🔍 VERIFICACIÓN DE ENTRENAMIENTO")
    print("=" * 40)
    
    checks = []
    
    # 1. Verificar modelos guardados
    model_files = glob.glob("models/saved_models/*.h5")
    metadata_file = "models/saved_models/model_metadata.json"
    
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        checks.append(("✅", f"Modelo encontrado: {os.path.basename(latest_model)}"))
        
        if os.path.exists(metadata_file):
            checks.append(("✅", "Metadata del modelo existe"))
        else:
            checks.append(("❌", "Metadata del modelo NO encontrada"))
    else:
        checks.append(("❌", "NO se encontró modelo entrenado"))
    
    # 2. Verificar checkpoints
    checkpoint_files = glob.glob("models/checkpoints/*.h5")
    if checkpoint_files:
        checks.append(("✅", f"Checkpoints encontrados: {len(checkpoint_files)}"))
    else:
        checks.append(("⚠️", "No se encontraron checkpoints"))
    
    # 3. Verificar resultados
    results_file = "results/evaluation_results.json"
    if os.path.exists(results_file):
        checks.append(("✅", "Archivo de resultados existe"))
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            accuracy = results.get('test_accuracy', 0)
            checks.append(("📊", f"Precisión en test: {accuracy:.4f} ({accuracy*100:.1f}%)"))
            
            if accuracy > 0.8:
                checks.append(("🎉", "Excelente precisión!"))
            elif accuracy > 0.6:
                checks.append(("👍", "Buena precisión"))
            else:
                checks.append(("⚠️", "Precisión baja - considera más entrenamiento"))
                
        except Exception as e:
            checks.append(("❌", f"Error leyendo resultados: {e}"))
    else:
        checks.append(("❌", "Archivo de resultados NO encontrado"))
    
    # 4. Verificar gráficos
    plots_dir = "results/plots"
    if os.path.exists(plots_dir):
        plot_files = os.listdir(plots_dir)
        if len(plot_files) > 0:
            checks.append(("✅", f"Gráficos generados: {len(plot_files)}"))
        else:
            checks.append(("⚠️", "Carpeta de gráficos vacía"))
    else:
        checks.append(("❌", "Carpeta de gráficos NO existe"))
    
    # 5. Verificar logs de entrenamiento
    log_files = glob.glob("results/training_log_*.csv")
    if log_files:
        checks.append(("✅", "Logs de entrenamiento encontrados"))
    else:
        checks.append(("⚠️", "No se encontraron logs de entrenamiento"))
    
    # 6. Verificar estructura de datos
    data_dirs = ["dataset/train", "dataset/validation", "dataset/test"]
    all_data_ok = True
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            if len(subdirs) == 5:  # 5 personajes
                checks.append(("✅", f"{dir_path}: {len(subdirs)} clases"))
            else:
                checks.append(("⚠️", f"{dir_path}: Solo {len(subdirs)} clases"))
                all_data_ok = False
        else:
            checks.append(("❌", f"{dir_path}: NO existe"))
            all_data_ok = False
    
    # Mostrar resultados
    print("\n📋 RESULTADOS DE VERIFICACIÓN:")
    for icon, message in checks:
        print(f"{icon} {message}")
    
    # Resumen final
    errors = sum(1 for icon, _ in checks if icon == "❌")
    warnings = sum(1 for icon, _ in checks if icon == "⚠️")
    successes = sum(1 for icon, _ in checks if icon == "✅")
    
    print(f"\n📊 RESUMEN:")
    print(f"✅ Verificaciones exitosas: {successes}")
    print(f"⚠️  Advertencias: {warnings}")
    print(f"❌ Errores: {errors}")
    
    if errors == 0:
        if warnings == 0:
            print(f"\n🎉 ¡TODO PERFECTO! El entrenamiento se completó exitosamente.")
        else:
            print(f"\n👍 Entrenamiento completado con algunas advertencias menores.")
        
        print(f"\n🚀 TU MODELO ESTÁ LISTO PARA:")
        print(f"   - Hacer predicciones de nuevas imágenes")
        print(f"   - Subir a Google Colab")
        print(f"   - Integrar en aplicaciones web")
        print(f"   - Deployment en producción")
        
    else:
        print(f"\n⚠️  Hay algunos problemas que necesitan atención.")
        print(f"   Revisa los errores marcados con ❌")
    
    return errors == 0

def show_model_info():
    """Mostrar información detallada del modelo"""
    
    metadata_file = "models/saved_models/model_metadata.json"
    
    if os.path.exists(metadata_file):
        print(f"\n📋 INFORMACIÓN DEL MODELO:")
        print("-" * 30)
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"🏷️  Clases: {', '.join(metadata.get('class_names', []))}")
        print(f"📏 Tamaño de imagen: {metadata.get('img_size', 'N/A')}")
        print(f"📅 Fecha de creación: {metadata.get('timestamp', 'N/A')}")
        print(f"📁 Archivo del modelo: {os.path.basename(metadata.get('model_path', 'N/A'))}")
    
    # Mostrar resultados de evaluación
    results_file = "results/evaluation_results.json"
    if os.path.exists(results_file):
        print(f"\n📊 MÉTRICAS DE EVALUACIÓN:")
        print("-" * 30)
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"🎯 Precisión: {results.get('test_accuracy', 0):.4f} ({results.get('test_accuracy', 0)*100:.1f}%)")
        print(f"🔝 Top-3 Accuracy: {results.get('test_top3_accuracy', 0):.4f}")
        print(f"📉 Loss: {results.get('test_loss', 0):.4f}")

def main():
    """Función principal"""
    
    training_success = verify_training_completion()
    
    if training_success:
        show_model_info()
        
        print(f"\n💡 PRÓXIMOS PASOS:")
        print(f"   1. Probar predicciones: py scripts/test_prediction.py")
        print(f"   2. Ver gráficos en: results/plots/")
        print(f"   3. Subir proyecto a Colab")
        print(f"   4. Crear aplicación de predicción")

if __name__ == "__main__":
    main()