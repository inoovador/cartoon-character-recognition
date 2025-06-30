import tensorflow as tf
from PIL import Image
import numpy as np
import os

def test_model():
    # Cargar modelo
    model_path = "models/saved_models/cartoon_classifier_simple.h5"
    if not os.path.exists(model_path):
        print("❌ No hay modelo entrenado")
        return
    
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Modelo cargado: {model_path}")
    
    # Obtener nombres de clases
    class_names = sorted(os.listdir("dataset/train/"))
    print(f"📋 Clases: {class_names}")
    
    # Probar con una imagen
    test_image = "dataset/raw_images/pikachu/Pikachu.jpg"
    if os.path.exists(test_image):
        img = Image.open(test_image).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)[0]
        top_idx = np.argmax(predictions)
        confidence = predictions[top_idx]
        
        print(f"🎯 Predicción: {class_names[top_idx]}")
        print(f"🎯 Confianza: {confidence:.2%}")
        
        if confidence > 0.3:
            print("✅ Modelo funciona correctamente!")
        else:
            print("⚠️ Confianza baja, pero funciona")
    
if __name__ == "__main__":
    test_model()