#!/usr/bin/env python3
"""
Script para descargar imágenes de personajes de cartoon/anime automáticamente
"""

import os
import requests
import time
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import pandas as pd
from tqdm import tqdm
import shutil

class CharacterImageDownloader:
    def __init__(self, base_path="dataset/raw_images"):
        self.base_path = base_path
        self.characters = {
            # Disney Characters
            "Mickey Mouse": "disney",
            "Minnie Mouse": "disney", 
            "Donald Duck": "disney",
            "Goofy": "disney",
            "Pluto Disney": "disney",
            "Simba Lion King": "disney",
            "Nemo": "disney",
            "Buzz Lightyear": "disney",
            "Woody Toy Story": "disney",
            "Elsa Frozen": "disney",
            "Olaf Frozen": "disney",
            "Moana": "disney",
            "Mulan": "disney",
            "Ariel Little Mermaid": "disney",
            "Belle Beauty Beast": "disney",
            
            # Anime Characters
            "Naruto Uzumaki": "anime",
            "Goku Dragon Ball": "anime",
            "Monkey D Luffy": "anime",
            "Pikachu": "anime",
            "Ash Ketchum": "anime",
            "Sailor Moon": "anime",
            "Vegeta": "anime",
            "Ichigo Bleach": "anime",
            "Edward Elric": "anime",
            "Natsu Dragneel": "anime",
            "Light Yagami": "anime",
            "Saitama One Punch": "anime",
            "Tanjiro Demon Slayer": "anime",
            "Eren Yeager": "anime",
            "Hinata Hyuga": "anime",
            
            # Cartoon Characters
            "Bugs Bunny": "cartoon",
            "Tom Cat": "cartoon",
            "Jerry Mouse": "cartoon", 
            "SpongeBob SquarePants": "cartoon",
            "Patrick Star": "cartoon",
            "Scooby Doo": "cartoon",
            "Shaggy Rogers": "cartoon",
            "Fred Flintstone": "cartoon",
            "Homer Simpson": "cartoon",
            "Bart Simpson": "cartoon",
            "Popeye": "cartoon",
            "Tweety Bird": "cartoon",
            "Sylvester Cat": "cartoon",
            "Daffy Duck": "cartoon",
            "Porky Pig": "cartoon"
        }
        
        # Crear carpetas para cada personaje
        for character in self.characters.keys():
            char_folder = character.replace(" ", "_").lower()
            os.makedirs(f"{self.base_path}/{char_folder}", exist_ok=True)
    
    def download_character_images(self, character_name, max_images=60):
        """Descargar imágenes de un personaje específico"""
        
        char_folder = character_name.replace(" ", "_").lower()
        output_dir = f"{self.base_path}/{char_folder}"
        
        print(f"🔽 Descargando imágenes de: {character_name}")
        
        try:
            # Usar Google Image Crawler
            google_crawler = GoogleImageCrawler(
                storage={'root_dir': output_dir},
                downloader_threads=4,
                parser_threads=2
            )
            
            # Términos de búsqueda más específicos
            search_terms = [
                f"{character_name} cartoon character",
                f"{character_name} anime character", 
                f"{character_name} face",
                f"{character_name} portrait"
            ]
            
            images_per_term = max_images // len(search_terms)
            
            for term in search_terms:
                try:
                    google_crawler.crawl(
                        keyword=term,
                        max_num=images_per_term,
                        file_idx_offset='auto'
                    )
                    time.sleep(2)  # Pausa para evitar rate limiting
                except Exception as e:
                    print(f"⚠️  Error con término '{term}': {e}")
                    continue
            
            # Verificar cuántas imágenes se descargaron
            downloaded_count = len([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ {character_name}: {downloaded_count} imágenes descargadas")
            
            return downloaded_count
            
        except Exception as e:
            print(f"❌ Error descargando {character_name}: {e}")
            return 0
    
    def download_all_characters(self):
        """Descargar imágenes de todos los personajes"""
        
        print("🚀 Iniciando descarga masiva de personajes...")
        
        results = []
        total_characters = len(self.characters)
        
        for i, (character, category) in enumerate(self.characters.items(), 1):
            print(f"\n📥 Progreso: {i}/{total_characters}")
            
            count = self.download_character_images(character)
            
            results.append({
                'character': character,
                'category': category, 
                'images_downloaded': count,
                'folder_name': character.replace(" ", "_").lower()
            })
            
            # Pausa entre personajes
            time.sleep(3)
        
        # Guardar resultados en CSV
        df = pd.DataFrame(results)
        df.to_csv('dataset/download_results.csv', index=False)
        
        print(f"\n🎉 Descarga completada!")
        print(f"📊 Total de personajes: {len(results)}")
        print(f"📊 Total de imágenes: {df['images_downloaded'].sum()}")
        
        return df
    
    def download_specific_characters(self, character_list, max_images=60):
        """Descargar solo personajes específicos"""
        
        results = []
        
        for character in character_list:
            if character in self.characters:
                count = self.download_character_images(character, max_images)
                results.append({
                    'character': character,
                    'category': self.characters[character],
                    'images_downloaded': count,
                    'folder_name': character.replace(" ", "_").lower()
                })
            else:
                print(f"⚠️  Personaje '{character}' no encontrado en la lista")
        
        return results

def main():
    """Función principal"""
    
    # Crear instancia del descargador
    downloader = CharacterImageDownloader()
    
    # Opción 1: Descargar todos los personajes
    print("¿Qué quieres hacer?")
    print("1. Descargar TODOS los personajes (45+ personajes)")
    print("2. Descargar personajes específicos")
    print("3. Descargar solo 10 personajes de prueba")
    
    choice = input("Elige una opción (1/2/3): ")
    
    if choice == "1":
        # Descargar todos
        results_df = downloader.download_all_characters()
        
    elif choice == "2":
        # Descargar específicos
        print("\nPersonajes disponibles:")
        for i, char in enumerate(downloader.characters.keys(), 1):
            print(f"{i}. {char}")
        
        selected = input("Escribe los nombres separados por coma: ").split(",")
        selected = [char.strip() for char in selected]
        
        results = downloader.download_specific_characters(selected)
        results_df = pd.DataFrame(results)
        
    elif choice == "3":
        # Solo 10 de prueba
        test_characters = [
            "Mickey Mouse", "Naruto Uzumaki", "SpongeBob SquarePants", 
            "Goku Dragon Ball", "Bugs Bunny", "Pikachu", 
            "Homer Simpson", "Elsa Frozen", "Tom Cat", "Sailor Moon"
        ]
        
        results = downloader.download_specific_characters(test_characters, max_images=30)
        results_df = pd.DataFrame(results)
    
    else:
        print("Opción no válida")
        return
    
    # Mostrar resumen
    if len(results_df) > 0:
        print("\n📈 RESUMEN DE DESCARGA:")
        print(results_df[['character', 'category', 'images_downloaded']])
        
        # Guardar metadata
        results_df.to_csv('dataset/metadata.csv', index=False)
        print("\n💾 Metadata guardada en 'dataset/metadata.csv'")

if __name__ == "__main__":
    main()