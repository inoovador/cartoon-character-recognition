#!/usr/bin/env python3
"""
Script rápido para crear splits train/validation/test
"""

import os
import shutil
import numpy as np
from tqdm import tqdm

def create_data_splits(processed_path="dataset/processed_images", 
                      train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Dividir dataset en train/validation/test"""
    
    # Verificar que suman 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
    
    train_dir = "dataset/train"
    val_dir = "dataset/validation" 
    test_dir = "dataset/test"
    
    # Limpiar directorios existentes
    for dir_path in [train_dir, val_dir, test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    character_folders = [f for f in os.listdir(processed_path) 
                       if os.path.isdir(os.path.join(processed_path, f))]
    
    print(f"📂 Creando splits para {len(character_folders)} personajes...")
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for character in tqdm(character_folders, desc="Creando splits"):
        char_path = os.path.join(processed_path, character)
        images = [f for f in os.listdir(char_path) if f.endswith('.jpg')]
        
        if len(images) < 10:  # Mínimo para hacer splits
            print(f"⚠️  Saltando {character}: solo {len(images)} imágenes")
            continue
        
        # Mezclar imágenes
        np.random.seed(42)  # Para reproducibilidad
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
        
        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)
        
        print(f"✅ {character}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    print(f"\n🎉 Splits creados exitosamente!")
    print(f"📊 Total imágenes:")
    print(f"   - Train: {total_train}")
    print(f"   - Validation: {total_val}")
    print(f"   - Test: {total_test}")

if __name__ == "__main__":
    create_data_splits()