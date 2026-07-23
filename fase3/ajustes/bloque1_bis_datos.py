# =====================================================================
# BLOQUE 1-BIS — REEMPLAZA LA CELDA 4 (configuración de entorno)
# ---------------------------------------------------------------------
# QUÉ SE CORRIGE
#   (a) La ruta original apuntaba al Drive del compañero:
#         .../Maestria/Redes_Neuronales_CD/Centinela_Proyecto_Fase1_y_Fase2
#       (nótese "Fase1_y_Fase2", sin la Fase 3). Desde otra cuenta no
#       existe → "NO ENCONTRADO".
#   (b) Los datos se leían DIRECTAMENTE desde Google Drive. Drive tiene
#       latencia alta por archivo, así que el benchmark del pipeline
#       medía sobre todo la I/O de Drive, no el diseño del pipeline.
#       Eso explica los 817 ms/batch del notebook original.
#       Ahora los datos se copian al disco local de Colab (/content),
#       que es SSD. El benchmark pasa a medir lo que dice medir.
#
# CÓMO USARLO
#   Modo A (recomendado) — subir datos_centinela_fase3.zip (15 MB):
#       MODO = 'subir'
#   Modo B — si ya subiste el zip a tu Drive:
#       MODO = 'drive_zip'  y ajusta RUTA_ZIP_DRIVE
#   Modo C — si la carpeta ya está descomprimida en tu Drive:
#       MODO = 'drive_carpeta'  y ajusta RUTA_BASE_DRIVE
# =====================================================================

MODO = 'subir'      # 'subir' | 'drive_zip' | 'drive_carpeta'

RUTA_ZIP_DRIVE   = '/content/drive/MyDrive/datos_centinela_fase3.zip'
RUTA_BASE_DRIVE  = '/content/drive/MyDrive/Maestria/Redes_Neuronales_CD/Centinela_Proyecto_Fase1_y_Fase2'

# ---------------------------------------------------------------------
import os, time, zipfile, shutil, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

DESTINO_LOCAL = '/content/datos_centinela'
RUTA_SALIDA   = '/content/modelos_fase3'
os.makedirs(RUTA_SALIDA, exist_ok=True)

def _descomprimir(ruta_zip, destino):
    t0 = time.time()
    os.makedirs(destino, exist_ok=True)
    with zipfile.ZipFile(ruta_zip) as z:
        z.extractall(destino)
    print(f'   Descomprimido en {time.time()-t0:.1f}s → {destino}')

if MODO == 'subir':
    from google.colab import files
    print('Selecciona datos_centinela_fase3.zip (15 MB) de tu computador...')
    subidos = files.upload()
    nombre_zip = list(subidos.keys())[0]
    print(f'   Recibido: {nombre_zip} ({os.path.getsize(nombre_zip)/1e6:.1f} MB)')
    _descomprimir(nombre_zip, DESTINO_LOCAL)
    RUTA_BASE = DESTINO_LOCAL

elif MODO == 'drive_zip':
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    assert os.path.exists(RUTA_ZIP_DRIVE), f'No existe: {RUTA_ZIP_DRIVE}'
    _descomprimir(RUTA_ZIP_DRIVE, DESTINO_LOCAL)
    RUTA_BASE = DESTINO_LOCAL

elif MODO == 'drive_carpeta':
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    assert os.path.exists(RUTA_BASE_DRIVE), f'No existe: {RUTA_BASE_DRIVE}'
    print('Copiando de Drive a disco local (mucho más rápido para entrenar)...')
    t0 = time.time()
    shutil.copytree(os.path.join(RUTA_BASE_DRIVE, 'Imagenes_agua'),
                    os.path.join(DESTINO_LOCAL, 'Imagenes_agua'), dirs_exist_ok=True)
    for sub, arch in [('data', 'water_dataset.mat')]:
        src = os.path.join(RUTA_BASE_DRIVE, sub, arch)
        if os.path.exists(src):
            os.makedirs(os.path.join(DESTINO_LOCAL, sub), exist_ok=True)
            shutil.copy(src, os.path.join(DESTINO_LOCAL, sub, arch))
    print(f'   Copiado en {time.time()-t0:.1f}s')
    RUTA_BASE = DESTINO_LOCAL

else:
    raise ValueError(f'MODO no reconocido: {MODO}')

# ── RUTAS DEL PROYECTO ───────────────────────────────────────────────
RUTA_IMG   = os.path.join(RUTA_BASE, 'Imagenes_agua', 'train')
RUTA_DATA  = os.path.join(RUTA_BASE, 'data', 'water_dataset.mat')
RUTA_PESOS = os.path.join(RUTA_BASE, 'fase2', 'pesos')

print('\n=== VERIFICACIÓN DE RUTAS ===')
todo_ok = True
for ruta, nombre, critico in [(RUTA_IMG,   'Imágenes',      True),
                              (RUTA_DATA,  'Dataset .mat',  False),
                              (RUTA_PESOS, 'Pesos Fase 2',  False)]:
    existe = os.path.exists(ruta)
    marca  = 'OK' if existe else ('FALTA (crítico)' if critico else 'falta (no requerido en Fase 3)')
    print(f'   {nombre:<15} {marca}')
    if critico and not existe:
        todo_ok = False

assert todo_ok, 'Faltan las imágenes: sin ellas la Fase 3 no corre.'

# Conteo por clase — la evidencia de que los datos llegaron completos
print('\n=== IMÁGENES POR CLASE ===')
total_imgs = 0
for clase in sorted(os.listdir(RUTA_IMG)):
    d = os.path.join(RUTA_IMG, clase)
    if os.path.isdir(d):
        n = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_imgs += n
        print(f'   {clase:<10} {n:>4}')
print(f'   {"TOTAL":<10} {total_imgs:>4}')
assert total_imgs == 512, f'Se esperaban 512 imágenes, hay {total_imgs}'
print('   Conteo correcto (512, igual que en Fase 2)')

# ── ENTORNO GPU ──────────────────────────────────────────────────────
import torch, torchvision
import tensorflow as tf

DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('\n=== ENTORNO GPU ===')
print(f'GPU disponible:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Modelo GPU:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM total:      {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'VRAM libre:      {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')
print(f'PyTorch:         {torch.__version__}')
print(f'TorchVision:     {torchvision.__version__}')
print(f'TensorFlow:      {tf.__version__}')
print(f'\nDatos en:        {RUTA_BASE}  (disco local, no Drive)')

get_ipython().system('nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv')
