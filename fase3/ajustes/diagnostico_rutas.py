# =====================================================================
# DIAGNÓSTICO DE RUTAS — correr en Colab ANTES del BLOQUE 1
# ---------------------------------------------------------------------
# Busca en tu Drive las carpetas del proyecto. El notebook original
# apuntaba a la ruta del compañero:
#   .../Maestria/Redes_Neuronales_CD/Centinela_Proyecto_Fase1_y_Fase2
# (ojo: "Fase1_y_Fase2", sin la Fase 3)
# =====================================================================

import os

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

RAIZ = '/content/drive/MyDrive'

print('=' * 68)
print('1 · ¿QUÉ HAY EN LA RAÍZ DE TU DRIVE?')
print('=' * 68)
for x in sorted(os.listdir(RAIZ))[:40]:
    tipo = 'DIR ' if os.path.isdir(os.path.join(RAIZ, x)) else 'file'
    print(f'  [{tipo}] {x}')

print()
print('=' * 68)
print('2 · BUSCANDO CARPETAS DEL PROYECTO (puede tardar ~1 min)')
print('=' * 68)

objetivos = ['centinela', 'imagenes_agua', 'water_dataset', 'bersih', 'redes_neuronales']
encontrados = {k: [] for k in objetivos}

for base, dirs, files in os.walk(RAIZ):
    # no descender en carpetas enormes irrelevantes
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    prof = base.replace(RAIZ, '').count(os.sep)
    if prof > 6:
        dirs[:] = []
        continue
    for nombre in dirs + files:
        low = nombre.lower()
        for k in objetivos:
            if k in low:
                encontrados[k].append(os.path.join(base, nombre))

for k, rutas in encontrados.items():
    print(f'\n"{k}" → {len(rutas)} coincidencia(s)')
    for r in rutas[:6]:
        print(f'    {r}')
    if len(rutas) > 6:
        print(f'    ... y {len(rutas)-6} más')

print()
print('=' * 68)
print('3 · VERIFICACIÓN DE LA RUTA QUE USA EL NOTEBOOK')
print('=' * 68)
RUTA_BASE = '/content/drive/MyDrive/Maestria/Redes_Neuronales_CD/Centinela_Proyecto_Fase1_y_Fase2'
print(f'RUTA_BASE = {RUTA_BASE}')
print(f'  ¿existe? {os.path.exists(RUTA_BASE)}')

# recorrer el camino tramo por tramo para ver dónde se rompe
partes, acum = RUTA_BASE.split('/'), ''
for p in partes:
    if not p:
        continue
    acum += '/' + p
    marca = 'OK ' if os.path.exists(acum) else '<<< SE ROMPE AQUÍ'
    print(f'  {marca} {acum}')
    if not os.path.exists(acum):
        padre = os.path.dirname(acum)
        if os.path.isdir(padre):
            print(f'       contenido real de {padre}:')
            for x in sorted(os.listdir(padre))[:20]:
                print(f'          - {x}')
        break

print()
print('=' * 68)
print('SIGUIENTE PASO')
print('=' * 68)
print('Si la sección 2 encontró "Imagenes_agua", copia su ruta y ajusta')
print('RUTA_BASE en el BLOQUE 1. Si no encontró nada, hay que subir los')
print('datos: ver ajustes/AJUSTES_Fase3.md o preguntar.')
