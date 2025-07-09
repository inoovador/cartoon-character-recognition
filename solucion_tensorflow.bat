@echo off
echo 🔧 SOLUCIONANDO PROBLEMAS DE TENSORFLOW
echo =========================================

echo.
echo 📋 Paso 1: Verificando configuración actual...
python --version
echo.
python -c "import platform; print('Arquitectura:', platform.architecture()[0]); print('Sistema:', platform.system())"

echo.
echo 📦 Paso 2: Limpiando cache de pip...
python -m pip cache purge

echo.
echo 🔄 Paso 3: Actualizando pip, setuptools y wheel...
python -m pip install --upgrade pip setuptools wheel

echo.
echo 🧠 Paso 4: Intentando instalar TensorFlow con diferentes enfoques...

echo.
echo "Intento 1: TensorFlow versión específica compatible..."
pip install tensorflow==2.10.0

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo "Intento 2: TensorFlow sin restricción de versión..."
    pip install tensorflow
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo "Intento 3: TensorFlow desde índice alternativo..."
    pip install --index-url https://pypi.org/simple/ tensorflow
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo "Intento 4: TensorFlow CPU específicamente..."
    pip install tensorflow-cpu
)

echo.
echo 🧪 Verificando instalación...
python -c "import tensorflow as tf; print('✅ TensorFlow instalado:', tf.__version__)" 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo ❌ TensorFlow no se pudo instalar
    echo.
    echo 💡 POSIBLES SOLUCIONES:
    echo 1. Actualizar Python a versión 3.8-3.11
    echo 2. Verificar que tienes Windows 64-bit
    echo 3. Usar entorno virtual
    echo 4. Instalar versión más antigua compatible
) else (
    echo.
    echo 🎉 ¡TensorFlow instalado correctamente!
)

pause