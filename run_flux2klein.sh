#!/bin/bash
#SBATCH --job-name=flux_2_std
#SBATCH --partition=gpu_h100            # Cola GPU H100 principal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96              # 96 cores (24 por GPU) para procesamiento intensivo
#SBATCH --mem=760000mb                  # Memoria máxima del nodo H100
#SBATCH --gres=gpu:4                    # 4 GPUs H100 en el mismo nodo
#SBATCH --time=72:00:00                 # Tiempo máximo asignado al job (72 horas)
#SBATCH --output=flux_2_%j.log          # Archivo de salida de logs

# ==============================================================================
# FLUX ControlNet (Canny) Product Image Standardization — SLURM Job Script
# ==============================================================================

# Limpiar rutas de Python heredadas del entorno del usuario
unset PYTHONPATH

# 1. Cargar CUDA 12.8 y Python
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

# 2. Variables del proyecto
WORKDIR=/pfs/data6/home/tu/tu_tu/tu_zxoxe46/austria_data
VENV=${WORKDIR}/.venv

# 3. Moverse al directorio del proyecto
cd ${WORKDIR}

# 4. Activar el entorno virtual
if [ ! -d "${VENV}" ]; then
    echo "Creando entorno virtual en ${VENV}..."
    python3 -m venv ${VENV}
fi

source ${VENV}/bin/activate

# Verificación e instalación silenciosa de dependencias
echo "Verificando dependencias en ${VENV}..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Debug: confirmar entorno correcto y GPU
echo "============================================================"
echo "Job ID:            $SLURM_JOB_ID"
echo "Node:              $SLURMD_NODENAME"
echo "Python location:   $(which python)"
echo "Python version:    $(python --version)"
echo "Torch CUDA:        $(python -c 'import torch; print(f\"GPUs disponibles: {torch.cuda.device_count()}\")')"
echo "============================================================"

# 5. Optimización de memoria VRAM de PyTorch para evitar fragmentación
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================================
# MODO DE EJECUCIÓN
# ==============================================================================
RUN_MODE="full"          # Cambiado a "full" para aprovechar las 72h
MAX_SAMPLES=5

if [ "$RUN_MODE" = "samples_n" ]; then
    echo "MODO: Test rápido -> Primeras ${MAX_SAMPLES} imágenes de samples/"
    # Si vas a usar torchrun para multi-gpu, el comando aquí cambiaría
    python main.py \
        --data-dir . \
        --samples-only \
        --max-samples ${MAX_SAMPLES}

elif [ "$RUN_MODE" = "samples_all" ]; then
    echo "MODO: Todas las imágenes de la carpeta samples/"
    python main.py \
        --data-dir . \
        --samples-only

elif [ "$RUN_MODE" = "full" ]; then
    echo "MODO: Producción -> Dataset completo (~107k imágenes)"
    python main.py \
        --data-dir .
fi

echo "============================================================"
echo "Job finalizado: $(date)"
echo "============================================================"