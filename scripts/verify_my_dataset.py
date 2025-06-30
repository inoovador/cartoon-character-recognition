# Crear scripts/verify_my_dataset.py
import os
from PIL import Image

def verify_downloaded_dataset():
    """Verificar dataset con TUS nombres de archivos"""
    
    characters = {
        "goku_dragon_ball": "Goku",
        "pikachu": "Pikachu", 
        "naruto_uzumaki": "Naruto",
        "mickey_mouse": "Mickey",
        "spongebob_squarepants": "Spongebob"
    }
    
    raw_path = "dataset/raw_images"
    total_images = 0
    
    print("🔍 VERIFICANDO TU DATASET PERSONALIZADO")
    print("=" * 50)
    
    for folder, prefix in characters.items():
        char_path = os.path.join(raw_path, folder)
        
        if not os.path.exists(char_path):
            print(f"❌ {folder}: Carpeta no existe")
            continue
        
        # Buscar archivos con tu patrón: Prefix.jpg, Prefix1.jpg, etc.
        expected_files = [f"{prefix}.jpg"] + [f"{prefix}{i}.jpg" for i in range(1, 10)]
        
        found_files = []
        valid_images = 0
        
        for expected_file in expected_files:
            file_path = os.path.join(char_path, expected_file)
            if os.path.exists(file_path):
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if width >= 100 and height >= 100:
                            found_files.append(expected_file)
                            valid_images += 1
                except Exception as e:
                    print(f"⚠️ Error en {expected_file}: {e}")
        
        total_images += valid_images
        status = "✅ Perfecto" if valid_images == 10 else f"⚠️ {valid_images}/10"
        
        print(f"{folder:25}: {valid_images:2d} imágenes - {status}")
        if valid_images < 10:
            missing = set(expected_files) - set(found_files)
            print(f"   📋 Faltan: {', '.join(missing)}")
    
    print("=" * 50)
    print(f"📊 TOTAL: {total_images}/50 imágenes")
    
    if total_images >= 40:
        print("🎉 ¡Dataset excelente para entrenamiento!")
        return True
    else:
        print("⚠️ Revisa las imágenes faltantes")
        return False

if __name__ == "__main__":
    verify_downloaded_dataset()