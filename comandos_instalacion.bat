@echo off
echo 🚀 INSTALANDO DEPENDENCIAS PARA CARTOON CHARACTER CLASSIFIER
echo ================================================================

echo.
echo 📦 Paso 1: Actualizando pip...
python -m pip install --upgrade pip

echo.
echo 🧠 Paso 2: Instalando TensorFlow...
pip install tensorflow>=2.10.0

echo.
echo 📊 Paso 3: Instalando librerías de ciencia de datos...
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install pandas>=1.4.0
pip install scikit-learn>=1.1.0

echo.
echo 🖼️ Paso 4: Instalando librerías de procesamiento de imágenes...
pip install pillow>=8.3.0
pip install opencv-python>=4.5.0

echo.
echo 📈 Paso 5: Instalando librerías adicionales...
pip install plotly>=5.0.0

echo.
echo 🏗️ Paso 6: Instalando EfficientNet...
pip install efficientnet

echo.
echo ✅ INSTALACIÓN COMPLETADA!
echo ================================================================
echo.
echo 🔍 Verificando instalaciones...
python -c "import tensorflow as tf; print('✅ TensorFlow:', tf.__version__)"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)"
python -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)"
python -c "import sklearn; print('✅ Scikit-learn:', sklearn.__version__)"

echo.
echo 🎉 ¡Listo para entrenar tu modelo!
echo Ejecuta: python train_simple_model.py
pause