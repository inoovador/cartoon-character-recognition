# 🎭 Cartoon Character Recognition

Un proyecto de Deep Learning para reconocer 44 personajes icónicos de anime, caricaturas y películas animadas usando Redes Neuronales Convolucionales (CNN).

## 🎯 Personajes Incluidos

### 🏠 Disney Clásicos
- Mickey Mouse, Minnie Mouse, Donald Duck, Goofy, Pluto
- Ariel (La Sirenita), Belle (La Bella y la Bestia), Elsa, Olaf (Frozen)
- Buzz Lightyear, Woody (Toy Story), Nemo, Moana, Mulan, Simba

### 📺 Caricaturas Clásicas  
- Bugs Bunny, Daffy Duck, Porky Pig, Tweety Bird, Sylvester Cat
- Tom Cat, Jerry Mouse, Fred Flintstone, Popeye
- Bart Simpson, Homer Simpson

### 🌟 Anime/Manga
- **Dragon Ball:** Goku, Vegeta
- **Naruto:** Naruto Uzumaki, Hinata Hyuga
- **One Piece:** Monkey D. Luffy
- **Attack on Titan:** Eren Yeager
- **Fullmetal Alchemist:** Edward Elric
- **Bleach:** Ichigo
- **Death Note:** Light Yagami
- **Fairy Tail:** Natsu Dragneel
- **One Punch Man:** Saitama
- **Demon Slayer:** Tanjiro
- **Sailor Moon**
- **Pokémon:** Ash Ketchum, Pikachu

### 🎪 Otros
- SpongeBob SquarePants, Patrick Star
- Scooby Doo, Shaggy Rogers

## 🚀 Instalación y Uso

### Prerrequisitos
```bash
Python 3.8+
Git
```

### Instalación
```bash
git clone https://github.com/tu-usuario/cartoon-character-recognition.git
cd cartoon-character-recognition
pip install -r requirements.txt
```

### Uso Completo

#### 1. Descargar Dataset
```bash
python scripts/download_images.py
```

#### 2. Preprocesar Imágenes
```bash
python scripts/preprocess_data.py
```

#### 3. Entrenar Modelo
```bash
python scripts/train_model.py
```

## 📊 Estructura del Proyecto

```
cartoon-character-recognition/
├── dataset/                    # Datos (no incluidos en repo)
│   ├── raw_images/            # Imágenes descargadas
│   ├── processed_images/      # Imágenes procesadas (224x224)
│   ├── train/                 # Conjunto de entrenamiento (70%)
│   ├── validation/            # Conjunto de validación (20%)
│   └── test/                  # Conjunto de prueba (10%)
├── models/                    # Modelos entrenados
│   ├── checkpoints/           # Checkpoints durante entrenamiento
│   └── saved_models/          # Modelos finales
├── results/                   # Resultados y métricas
│   ├── plots/                # Gráficos de entrenamiento
│   └── reports/              # Reportes de evaluación
├── scripts/                   # Scripts principales
│   ├── download_images.py    # Descarga de imágenes
│   ├── preprocess_data.py    # Limpieza y preprocesamiento  
│   └── train_model.py        # Entrenamiento del modelo
├── src/                      # Código fuente modular
│   ├── data/                 # Utilidades de datos
│   ├── models/               # Arquitecturas de modelos
│   └── utils/                # Funciones auxiliares
└── notebooks/                # Análisis exploratorio
```

## 🧠 Modelos Disponibles

### Modelo Básico (CNN desde cero)
- 4 bloques convolucionales
- Batch Normalization y Dropout
- ~1M parámetros
- Tiempo de entrenamiento: ~2-3 horas

### Transfer Learning (MobileNetV2)
- Modelo preentrenado en ImageNet
- Fine-tuning para personajes
- ~2.3M parámetros
- Tiempo de entrenamiento: ~1 hora

## 📈 Rendimiento Esperado

- **Accuracy:** 85-95% (dependiendo del modelo)
- **Top-5 Accuracy:** >95%
- **Tamaño de imagen:** 224x224 píxeles
- **Clases:** 44 personajes

## 🔧 Características Técnicas

### Preprocesamiento
- ✅ Eliminación de duplicados (hash MD5)
- ✅ Validación de imágenes corruptas
- ✅ Redimensionado a 224x224
- ✅ Normalización RGB
- ✅ División automática train/val/test

### Aumento de Datos
- ✅ Rotación (±20°)
- ✅ Desplazamiento horizontal/vertical
- ✅ Zoom aleatorio
- ✅ Volteo horizontal

### Entrenamiento
- ✅ Early Stopping
- ✅ Reducción de Learning Rate
- ✅ Checkpoints automáticos
- ✅ Métricas detalladas

## 📊 Resultados y Visualizaciones

El proyecto genera automáticamente:
- 📈 Curvas de entrenamiento (accuracy/loss)
- 🔥 Matriz de confusión
- 📋 Reporte de clasificación por personaje
- 💾 Historial en formato JSON

## 🛠️ Personalización

### Agregar Nuevos Personajes
1. Crear carpeta en `dataset/raw_images/nuevo_personaje/`
2. Agregar imágenes (mínimo 15)
3. Ejecutar preprocesamiento
4. Reentrenar modelo

### Ajustar Hiperparámetros
Modificar en `scripts/train_model.py`:
- Learning rate
- Batch size
- Epochs
- Arquitectura del modelo

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- Datasets de imágenes de personajes
- TensorFlow/Keras community
- OpenCV contributors

---

**¿Preparado para entrenar tu propio clasificador de personajes? ¡Ejecuta los scripts y disfruta! 🚀**