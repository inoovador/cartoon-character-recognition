#!/usr/bin/env python3
"""
Script para limpiar y preprocesar el dataset de personajes
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import shutil
from tqdm import tqdm
import hashlib

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DatasetPreprocessor:
    def __init__(self, raw_path="dataset/raw_images", processed_path="dataset/processed_images"):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.target_size = (224, 224)  # Tamaño estándar para CNN
        self.min_images_per_character = 15  # Mínimo para incluir personaje
        
        # Crear carpeta procesada
        os.makedirs(self.processed_path, exist_ok=True)
    
    def is_valid_image(self, image_path):
        """Verificar si la imagen es válida"""
        try:
            # Intentar abrir con PIL
            with Image.open(image_path) as img:
                img.verify()
            
            # Intentar abrir con OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            # Verificar dimensiones mínimas
            height, width = img.shape[:2]
            if height < 50 or width < 50:
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Imagen inválida {image_path}: {e}")
            return False
    
    def get_image_hash(self, image_path):
        """Obtener hash de imagen para detectar duplicados"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def remove_duplicates(self, character_folder):
        """Eliminar imágenes duplicadas en una carpeta"""
        
        image_files = [f for f in os.listdir(character_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        if len(image_files) == 0:
            return 0
        
        print(f"🔍 Verificando duplicados en {os.path.basename(character_folder)}...")
        
        hashes = {}
        removed_count = 0
        
        for img_file in tqdm(image_files, desc="Checking duplicates"):
            img_path = os.path.join(character_folder, img_file)
            
            # Verificar si es válida
            if not self.is_valid_image(img_path):
                os.remove(img_path)
                removed_count += 1
                continue
            
            # Verificar duplicados
            img_hash = self.get_image_hash(img_path)
            if img_hash:
                if img_hash in hashes:
                    # Duplicado encontrado
                    os.remove(img_path)
                    removed_count += 1
                else:
                    hashes[img_hash] = img_path
        
        remaining = len(os.listdir(character_folder))
        print(f"✅ {os.path.basename(character_folder)}: {removed_count} eliminadas, {remaining} válidas")
        
        return remaining
    
    def resize_and_convert(self, input_path, output_path):
        """Redimensionar imagen y convertir a formato estándar"""
        try:
            # Abrir imagen
            img = Image.open(input_path)
            
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionar manteniendo aspecto
            img.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            
            # Crear imagen final con fondo blanco
            final_img = Image.new('RGB', self.target_size, (255, 255, 255))
            
            # Centrar imagen
            x = (self.target_size[0] - img.width) // 2
            y = (self.target_size[1] - img.height) // 2
            final_img.paste(img, (x, y))
            
            # Guardar como JPG
            final_img.save(output_path, 'JPEG', quality=95)
            return True
            
        except Exception as e:
            print(f"❌ Error procesando {input_path}: {e}")
            return False
    
    def process_character_folder(self, character_name):
        """Procesar carpeta de un personaje específico"""
        
        input_folder = os.path.join(self.raw_path, character_name)
        output_folder = os.path.join(self.processed_path, character_name)
        
        if not os.path.exists(input_folder):
            print(f"⚠️  Carpeta no encontrada: {input_folder}")
            return 0
        
        # Crear carpeta de salida
        os.makedirs(output_folder, exist_ok=True)
        
        # Limpiar duplicados primero
        valid_count = self.remove_duplicates(input_folder)
        
        if valid_count < self.min_images_per_character:
            print(f"⚠️  {character_name}: Solo {valid_count} imágenes válidas (mínimo {self.min_images_per_character})")
            return 0
        
        # Procesar imágenes válidas
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        processed_count = 0
        
        print(f"🖼️  Procesando {len(image_files)} imágenes de {character_name}...")
        
        for i, img_file in enumerate(tqdm(image_files, desc=f"Processing {character_name}")):
            input_path = os.path.join(input_folder, img_file)
            output_filename = f"{character_name}_{i+1:03d}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            if self.resize_and_convert(input_path, output_path):
                processed_count += 1
        
        print(f"✅ {character_name}: {processed_count} imágenes procesadas")
        return processed_count
    
    def process_all_characters(self):
        """Procesar todas las carpetas de personajes"""
        
        if not os.path.exists(self.raw_path):
            print(f"❌ Carpeta no encontrada: {self.raw_path}")
            return
        
        character_folders = [f for f in os.listdir(self.raw_path) 
                           if os.path.isdir(os.path.join(self.raw_path, f))]
        
        if len(character_folders) == 0:
            print("❌ No se encontraron carpetas de personajes")
            return
        
        print(f"🚀 Procesando {len(character_folders)} personajes...")
        
        results = []
        
        for character in character_folders:
            processed_count = self.process_character_folder(character)
            
            results.append({
                'character': character,
                'processed_images': processed_count,
                'status': 'success' if processed_count >= self.min_images_per_character else 'insufficient'
            })
        
        # Guardar resultados
        df = pd.DataFrame(results)
        df.to_csv('dataset/preprocessing_results.csv', index=False)
        
        # Resumen
        successful = df[df['status'] == 'success']
        total_images = successful['processed_images'].sum()
        
        print(f"\n📊 RESUMEN DE PROCESAMIENTO:")
        print(f"✅ Personajes válidos: {len(successful)}")
        print(f"📸 Total imágenes procesadas: {total_images}")
        print(f"⚠️  Personajes descartados: {len(df) - len(successful)}")
        
        if len(successful) > 0:
            print(f"\n🎯 Personajes exitosos:")
            for _, row in successful.iterrows():
                print(f"   - {row['character']}: {row['processed_images']} imágenes")
        
        return df
    
    def create_data_splits(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Dividir dataset en train/validation/test"""
        
        # Verificar que suman 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        train_dir = "dataset/train"
        val_dir = "dataset/validation" 
        test_dir = "dataset/test"
        
        # Crear directorios
        for dir_path in [train_dir, val_dir, test_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        character_folders = [f for f in os.listdir(self.processed_path) 
                           if os.path.isdir(os.path.join(self.processed_path, f))]
        
        print(f"📂 Creando splits para {len(character_folders)} personajes...")
        
        for character in character_folders:
            char_path = os.path.join(self.processed_path, character)
            images = [f for f in os.listdir(char_path) if f.endswith('.jpg')]
            
            if len(images) < 10:  # Mínimo para hacer splits
                continue
            
            # Mezclar imágenes
            np.random.shuffle(images)
            
            # Calcular splits
            n_train = int(len(images) * train_ratio)
            n_val = int(len(images) * val_ratio)
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Crear carpetas por personaje
            for split_dir in [train_dir, val_dir, test_dir]:
                os.makedirs(os.path.join(split_dir, character), exist_ok=True)
            
            # Copiar imágenes
            for img in train_images:
                shutil.copy2(os.path.join(char_path, img), 
                           os.path.join(train_dir, character, img))
            
            for img in val_images:
                shutil.copy2(os.path.join(char_path, img), 
                           os.path.join(val_dir, character, img))
            
            for img in test_images:
                shutil.copy2(os.path.join(char_path, img), 
                           os.path.join(test_dir, character, img))
            
            print(f"✅ {character}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
        print(f"\n🎉 Splits creados exitosamente!")

def main():
    """Función principal"""
    
    preprocessor = DatasetPreprocessor()
    
    print("🧹 LIMPIEZA Y PREPROCESAMIENTO DEL DATASET")
    print("=" * 50)
    
    # Procesar todas las imágenes
    results_df = preprocessor.process_all_characters()
    
    if results_df is not None and len(results_df) > 0:
        # Crear splits de datos
        print(f"\n📂 Creando splits train/validation/test...")
        preprocessor.create_data_splits()
        
        print(f"\n✅ ¡Preprocesamiento completado!")
        print(f"📁 Imágenes procesadas en: dataset/processed_images/")
        print(f"📁 Splits en: dataset/train/, dataset/validation/, dataset/test/")
    else:
        print(f"❌ No se pudo procesar ningún personaje")

if __name__ == "__main__":
    main()