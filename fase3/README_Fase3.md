# Proyecto Centinela — Fase 3

**Implementación y Herramientas Profesionales de Deep Learning**
Universidad Santo Tomás · Maestría en Ciencia de Datos
Profundización I: Redes Neuronales — Deep Learning (EA-N-F-004)
Docente: Javier Mauricio Sierra · 2026

---

## Qué es esta fase

La Fase 3 **industrializa** el sistema construido en las fases 1 y 2. No se cambia el
problema ni las fuentes de datos: se toma el modelo que ya existe y se lo prepara para
funcionar **offline en una tablet en campo**, donde la conectividad es intermitente.

El rol cambia de científico de datos a **ingeniero de MLOps**.

**Escenario:** Junta Administradora de Acueducto Rural (JAAR) que necesita detectar
contaminación del agua sin depender de laboratorios ni de internet.

---

## Estructura de la entrega

```
fase3/
├── README_Fase3.md                     ← este archivo
├── Centinela_Fase3.ipynb               ← notebook ejecutado en Colab, con salidas
├── docs/
│   ├── Informe_Tecnico_Fase3.pdf       ← informe de despliegue (máx. 4 páginas)
│   ├── Bitacora_IA_Fase3.pdf           ← bitácora de uso de IA
│   └── AJUSTES_Fase3.md                ← registro de correcciones aplicadas
├── modelos/
│   ├── Centinela_Fase3_ResNet18.pth    ← PyTorch nativo
│   ├── Centinela_Fase3_ResNet18.onnx   ← ONNX, archivo único, verificado
│   ├── Centinela_Fase3_ResNet.keras    ← implementación Keras equivalente
│   ├── mejor_resnet18.pth              ← mejor checkpoint por pérdida de validación
│   ├── modelo_fp32.pth                 ← base para la comparativa Edge
│   ├── modelo_int8_dinamico.pth        ← cuantización dinámica (contraste)
│   ├── modelo_int8_estatico.pth        ← cuantización estática PTQ ← la que se despliega
│   ├── checkpoint_epoca_05.pth
│   ├── checkpoint_epoca_10.pth
│   └── checkpoint_epoca_15.pth
├── figuras/
│   ├── comparativa_pipelines.png
│   ├── augmentation_visual.png
│   ├── curvas_entrenamiento_f3.png
│   ├── comparativa_fp32_fp16.png
│   ├── matriz_confusion_f3.png
│   └── optimizacion_edge.png
└── cifras_informe.json                 ← única fuente de verdad de los números
```

> **`cifras_informe.json`** lo genera el último bloque del notebook. **Ningún número
> del informe se escribe a mano:** si no aparece en ese archivo, no se reporta. Es la
> regla que adoptamos tras la retroalimentación de la Fase 2.

---

## Datos

- **Rama visual:** 512 fotografías propias de agua, tres clases —
  `Bersih` / Limpia (311), `Keruh` / Turbia (103), `Kotor` / Contaminada (98).
- **Rama temporal:** `data/water_dataset.mat`, 11 rasgos normalizados a [0, 1]
  (UCI Machine Learning Repository — fuente admitida por el catálogo del curso).
- **Política de datos:** no se usan fuentes de Kaggle.

### Partición

Partición **estratificada** 70 / 15 / 15 sobre índices fijos con `random_state=42`,
calculada una sola vez y reutilizada en todo el notebook. El notebook incluye un
`assert` que comprueba que train, val y test son disjuntos, y su salida se conserva.

| Conjunto | n | Transformaciones |
|---|---|---|
| Train | ~358 | Augmentation en vivo (5 transformaciones) |
| Val | ~76 | Solo resize + normalización |
| Test | ~78 | Solo resize + normalización |

---

## Cómo reproducir

1. Abrir `Centinela_Fase3.ipynb` en Google Colab.
2. Entorno de ejecución → Cambiar tipo → **GPU T4**.
3. Ajustar `RUTA_BASE` a la ubicación del proyecto en Drive.
4. Ejecutar → Ejecutar todas las celdas.

**Entorno verificado:** Tesla T4 (15,6 GB) · PyTorch 2.11.0+cu128 · TensorFlow 2.20.0 ·
CUDA 13.0 · Python 3.12.

Tiempo aproximado de corrida completa: ~15 minutos (incluye los experimentos
comparativos FP32/FP16 y el A/B de augmentation).

---

## Los nueve bloques del notebook

| # | Bloque | Qué produce |
|---|---|---|
| 0 | Instalación | `onnx`, `onnxruntime`, `onnxscript`, `tensorflow` |
| 1 | Entorno GPU | GPU, VRAM, cuota, versiones (`nvidia-smi`) |
| 2 | Pipeline de datos | tf.data vs DataLoader, comparativa en época 1 y 2 |
| 3 | Data Augmentation | 5 transformaciones + grilla visual + A/B con y sin |
| 4 | Entrenamiento GPU | ResNet-18 + AMP FP16 + checkpointing + FP32/FP16 medido + CUDA OOM |
| 5 | Curvas | Pérdida y accuracy, FP32 vs FP16 |
| 6 | Evaluación | Matriz de confusión + accuracy y ROC-AUC **con IC95%** |
| 7 | Optimización Edge | Cuantización dinámica vs estática, tabla antes/después |
| 8 | Exportación | `.pth`, `.onnx` (ruta estable, archivo único), `.keras` + verificación |
| 9 | Ruta de despliegue | Arquitectura dual y despliegue responsable |
| 10 | Cifras | `cifras_informe.json` |

---

## Decisiones técnicas y su justificación

### Pipeline

Se implementó el mismo pipeline en ambos frameworks. La clave es el **prefetch**:
solapar la preparación de datos en CPU con el cómputo en GPU, de modo que
`t_época ≈ max(t_CPU, t_GPU)` en lugar de `t_CPU + t_GPU`.

- **PyTorch:** `num_workers=2`, `pin_memory=True`, `prefetch_factor=2`.
- **tf.data:** `.shuffle()` + `.map(AUTOTUNE)` + `.cache()` + `.batch()` + `.prefetch(AUTOTUNE)`.

**Sobre la medición.** Ambos pipelines llevan el **mismo `.shuffle()`** con buffer igual
al dataset; lo único que varía es el paralelismo, el cache y el prefetch. Y se mide en
**época 1 y época 2 por separado**: el cache solo tiene efecto a partir de la segunda
pasada, cuando ya está lleno. Comparar en la primera época mide el llenado del cache,
no su beneficio.

### Data Augmentation

Cinco transformaciones, elegidas por su semántica en este dominio:

| # | Transformación | Por qué |
|---|---|---|
| 1 | `RandomCrop(224)` tras resize a 256 | Encuadres distintos del mismo cuerpo de agua |
| 2 | `RandomHorizontalFlip(p=0.5)` | El agua no tiene orientación izquierda/derecha |
| 3 | `RandomRotation(15°)` | La tablet no siempre se sostiene nivelada |
| 4 | `ColorJitter(brillo .2, contraste .2, sat .1, tono .05)` | Luz de mañana vs mediodía |
| 5 | `GaussianBlur(k=3)` | Enfoque imperfecto en campo |

**Rechazadas y por qué:**

- `RandomVerticalFlip` — una foto de agua invertida no tiene sentido físico.
- `ColorJitter` fuerte (`saturation=0.3`, `hue=0.1`) — el color **es** la señal
  diagnóstica: alterar el tono puede volver el agua turbia (café) en aparentemente
  limpia (azul). Se redujo a `saturation=0.1`, `hue=0.05`.

El aumento se aplica **en vivo sobre las imágenes** en cada época, no sobre features
precomputadas. Esto corrige un señalamiento directo de la Fase 2.

### Entrenamiento

ResNet-18 con transfer learning desde ImageNet, precisión mixta FP16 con `GradScaler`,
`ReduceLROnPlateau`, early stopping (paciencia 8) y checkpointing cada 5 épocas.

**Por qué ResNet-18 y no ResNet-50:** 11,2 M parámetros frente a 25,6 M. El destino es
una tablet sin GPU; el modelo más liviano que resuelve la tarea es el correcto.

**Los checkpoints guardan `state_dict` + estado del optimizador + `mejor_loss_val`**, no
solo los pesos. Sin el optimizador no se puede retomar un entrenamiento correctamente:
Adam pierde sus momentos y el reinicio introduce un salto en la trayectoria.

**FP32 vs FP16 se mide, no se estima.** Se entrenan 5 épocas en cada precisión con la
misma semilla y se registran VRAM pico y segundos por época. El ahorro de VRAM es
menor al 50% porque AMP mantiene los pesos maestros en FP32; el ahorro se concentra
en las activaciones.

### Optimización Edge

Se aplicaron **dos** técnicas y se comparan:

1. **Cuantización dinámica** (`quantize_dynamic`, `qconfig_spec={nn.Linear}`) — se
   conserva como contraste documentado. ResNet-18 tiene **una sola** capa `Linear`
   (512→3 = 1.539 parámetros, el **0,014%** del modelo); todo lo demás son `Conv2d`,
   que esta técnica no alcanza. Por eso **no reduce el tamaño**. El notebook imprime
   esa fracción explícitamente.
2. **Cuantización estática PTQ** (`fbgemm`, con fusión Conv+BN+ReLU y calibración sobre
   128 imágenes **de train**) — sí cuantiza las convoluciones. Es la que se despliega.

Las tres columnas de la tabla (FP32 / INT8 dinámico / INT8 estático) se miden sobre
**el mismo conjunto de test**, con las mismas imágenes y el mismo protocolo. La latencia
se reporta como **mediana** de 30 corridas tras 10 de calentamiento.

### Exportación

| Formato | Uso |
|---|---|
| `.pth` | Inferencia y re-entrenamiento en PyTorch |
| `.onnx` | Puente portable entre plataformas — **archivo único** |
| `.keras` | Punto de partida para la conversión a LiteRT (`.tflite`) |

**Ruta estable explícita.** Se exporta con `dynamo=False`. En PyTorch 2.11 el exportador
dynamo es el **predeterminado**: no pasar `dynamo=True` no equivale a usar la ruta
legacy. Hay que pedirla explícitamente.

**Archivo único.** Por defecto el exportador deja los pesos en un sidecar
`.onnx.data`; el `.onnx` queda en ~90 KB y **no carga si se entrega solo**. Se fuerza
`save_as_external_data=False`.

**Verificación de consistencia.** `np.allclose(atol=1e-4)` sobre **logits** (no sobre
softmax, que comprime diferencias y hace la prueba más permisiva), evaluada tanto con
un tensor dummy como con **un batch de imágenes reales de test**, más el porcentaje de
acuerdo entre predicciones. Un modelo que predice distinto tras exportarse no es
desplegable.

**`.pth` y `.keras` son dos implementaciones equivalentes** (ResNet-18 en PyTorch,
ResNet50V2 en Keras), no el mismo objeto serializado dos veces. Ambos se entrenan; sus
accuracies no tienen por qué coincidir y se reportan por separado.

---

## Ruta de despliegue

### Arquitectura dual

**Rama visual (CNN) — offline en la tablet**
Modelo ResNet-18 cuantizado INT8 → LiteRT (`.tflite`) desde el modelo Keras.
Flujo: foto del río → clasificación → Limpia / Turbia / Contaminada.
**Funciona sin internet.**

**Rama temporal (GRU) — API REST en la nube**
Modelo GRU de la Fase 2 servido con ONNX Runtime.
Flujo: 14 días de mediciones de pH → API REST → probabilidad de riesgo.

**Por qué separarlas:** los modelos recurrentes tienen operadores que no siempre están
soportados en LiteRT y cuyo comportamiento varía entre versiones del runtime. Servir la
rama temporal en la nube aprovecha la conectividad intermitente sin sacrificar la
capacidad de operar offline con la rama visual.

### Flujo por conectividad

- **Sin internet:** solo CNN en la tablet → alerta visual inmediata.
- **Con internet:** CNN (tablet) + GRU (nube) → señal complementaria con horizonte temporal.

> **Advertencia honesta.** La *fusión* de ambas ramas, tal como se implementó en la
> Fase 2, **no está validada**: la clave de alineación quedó constante y la modalidad
> visual no llegó a participar del modelo. Hasta rehacer esa ablación, el sistema se
> documenta como **dos señales que se presentan al operador por separado**, no como un
> modelo fusionado. Ver `AJUSTES_Fase3.md`.

### Despliegue responsable

**Transparencia.** La JAAR es informada de que el sistema usa IA entrenada con 512
imágenes y de que sus predicciones son orientativas. La interfaz declara las
limitaciones del modelo.

**Beneficio social.** Detección proactiva sin depender de laboratorios ni de
conectividad permanente. Reduce el tiempo de respuesta de días a segundos en
comunidades rurales con acceso limitado a servicios técnicos.

**Mitigación de errores.** Los costos son asimétricos:

- *Falso positivo* (alarma incorrecta) → monitoreo adicional. Costo bajo, efecto conservador.
- *Falso negativo* (no detectar riesgo real) → puede comprometer la salud de la comunidad. **Costo alto.**

Por esa asimetría se priorizan umbrales conservadores y se prioriza el **recall** sobre
la precisión. El sistema **complementa** la decisión humana del operador, no la
reemplaza.

---

## Relación con las fases anteriores

| Fase | Aporte | Qué se reutiliza aquí |
|---|---|---|
| 1 | MLP sobre datos tabulares | Marco ético y propuesta de datos |
| 2 | CNN (visual) + LSTM/GRU (temporal) + fusión | Rama visual industrializada; GRU para la API |
| **3** | **Pipeline, GPU, Edge, despliegue** | **Cierra el proyecto integrador** |

---

## Uso de IA

El uso de IA generativa está permitido y documentado. La herramienta utilizada fue
Claude (Anthropic). Cada interacción relevante está registrada en `Bitacora_IA_Fase3.pdf`
con su decisión — **Aceptada / Modificada / Rechazada** — y su justificación técnica.

Las decisiones sobre arquitectura, hiperparámetros e interpretación de resultados
fueron tomadas por el equipo. Ningún bloque se incluyó sin comprenderlo.

---

## Referencias

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR)*, 770–778. https://doi.org/10.1109/CVPR.2016.90

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., &
Kalenichenko, D. (2018). Quantization and training of neural networks for efficient
integer-arithmetic-only inference. *CVPR*, 2704–2713. https://arxiv.org/abs/1712.05877

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., & Wu, H.
(2018). Mixed precision training. *International Conference on Learning Representations
(ICLR)*. https://arxiv.org/abs/1710.03740

PyTorch Team. (2024). *torch.onnx.export — PyTorch documentation*.
https://pytorch.org/docs/stable/onnx.html

TensorFlow Authors. (2023). *Better performance with the tf.data API*.
https://www.tensorflow.org/guide/data_performance

Google AI Edge. (2024). *LiteRT (formerly TensorFlow Lite) overview*.
https://ai.google.dev/edge/litert

Zhao, L. (2019). *Water Quality Prediction* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.1145/3339823

---

*Universidad Santo Tomás · Maestría en Ciencia de Datos · 2026*
