# Modelos entrenados — Fase 3

Por el límite de GitHub (máximo 100 MB por archivo), los pesos completos de los
modelos **no** están en este repositorio. En esta carpeta solo se incluye:

| Archivo | Tamaño | Descripción |
|---|---|---|
| `modelo_int8_estatico.pth` | ~11 MB | ResNet18 cuantizado (INT8 estático), listo para inferencia ligera |

## Modelos completos (disponibles en Google Drive)

| Archivo | Tamaño | Descripción |
|---|---|---|
| `Centinela_Fase3_ResNet.keras` | 270 MB | Modelo Keras/TensorFlow completo |
| `Centinela_Fase3_ResNet18.pth` | 43 MB | Pesos finales ResNet18 (PyTorch) |
| `Centinela_Fase3_ResNet18.onnx` | 43 MB | Exportación ONNX |
| `mejor_resnet18.pth` | 43 MB | Mejor checkpoint por validación |
| `modelo_fp32.pth` | 43 MB | Modelo en precisión FP32 |
| `modelo_int8_dinamico.pth` | 43 MB | Cuantización INT8 dinámica |
| `checkpoint_epoca_05.pth` / `checkpoint_epoca_10.pth` | 128 MB c/u | Checkpoints de entrenamiento |

> **Enlace de descarga:** _(pegar aquí el enlace de la carpeta
> `Proyecto_Centinela/03_Fase3/modelos` de Google Drive una vez subida)_
