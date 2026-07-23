# Registro de ajustes — Fase 3

Documento de trabajo. Consolida (a) las correcciones que pidió el docente en la
retroalimentación de la Fase 2 y (b) los defectos detectados al auditar el notebook,
el informe y la bitácora de la Fase 3 antes de entregar.

**Este documento alimenta la bitácora de IA.** Los hallazgos 3 y 4 son auditorías
propias sobre código generado con asistencia de IA: son exactamente lo que el nivel
Estratégico del criterio 5 pide («auditoría con un fallo no trivial corregido»).

---

## Parte A · Correcciones pedidas por el docente en la Fase 2

Nota obtenida en Fase 2: **4,2 / 5,0 — nivel Autónomo**.

| # | Corrección pedida | Estado | Dónde queda resuelto |
|---|---|---|---|
| 4 | Activar de verdad el data augmentation, o no declararlo | ✅ **Resuelta** | Se aplica en vivo sobre imágenes en `loader_train`, no sobre features precomputadas |
| 7 | Cuaderno coherente, con salidas visibles y pesos | ✅ **Resuelta** | Notebook corrido con las 23 celdas y sus salidas |
| 5 | Reportar incertidumbre en vez de cuatro decimales | ✅ **Resuelta ahora** | BLOQUE 6-BIS: IC95% por bootstrap para accuracy y ROC-AUC |
| 7b | README, formato PDF, marcadores pendientes | ✅ **Resuelta ahora** | `README_Fase3.md` creado; informe a PDF de 4 páginas |
| 6 | Respaldar la decisión de Fine-Tuning con el experimento | 🟡 **Parcial** | Se hace fine-tuning completo, pero no se corre la comparación FE vs FT |
| 1 | Corregir la clave de alineación (`×10`) y reejecutar la ablación | ❌ **Pendiente** | Es de la Rama C (Fase 2); no se toca en Fase 3 |
| 2 | Entregar la Rama C con salidas y pesos | ❌ **Pendiente** | Ídem |
| 3 | Reinterpretar la «ganancia» de la fusión | 🟡 **Mitigada** | Se retira la afirmación de fusión validada del informe y del README |

### Sobre las tres pendientes

Corresponden a la Rama C de la Fase 2 y **su nota ya está cerrada**. No entran en la
rúbrica de la Fase 3. Pero el docente anunció que las preguntará en la **sustentación
oral** y calificó dos de ellas como «las decisivas».

Decisión tomada: **no afirmar en la Fase 3 nada que dependa de la fusión no validada.**
El informe y el README ahora describen el sistema como **dos señales complementarias
que se presentan al operador por separado**, no como un modelo fusionado. Es honesto y
además blinda la entrega.

---

## Parte B · Defectos detectados al auditar la Fase 3

### 🔴 Hallazgo 1 — El accuracy INT8 de «100%» no medía nada

**Severidad: crítica.** Es el mismo error que costó la nota en la Fase 2: un número
reportado que no mide lo que dice medir.

Código original (celda 16):

```python
loader_test_cpu = DataLoader(
    datasets.ImageFolder(root=RUTA_IMG, transform=transform_val),  # las 512 imágenes
    batch_size=32, shuffle=False)                                   # sin barajar
...
for i, (imgs, etiq) in enumerate(loader_test_cpu):
    if i > 3: break     # 4 batches = 128 imágenes
```

`ImageFolder` sin barajar recorre las clases en orden alfabético, y `Bersih` (Limpia)
tiene **311 imágenes**. Por tanto esas 128 imágenes son:

1. **Todas de una sola clase** — clasificarlas bien es trivial para un modelo de 3 clases.
2. **Todas del conjunto de entrenamiento** — no se aplicó ningún split. El modelo ya las vio.

De ahí el 100%. El informe lo llevó a la tabla como **«+2,6 pp»** de mejora y el resumen
del notebook imprimió `Accuracy: 97.4% (FP32) → 100.0% (INT8)`.

**La cuantización no puede mejorar el accuracy.** Un resultado que contradice la teoría
es una señal de error de medición, no un hallazgo.

**Corrección:** BLOQUE 7-BIS evalúa FP32 e INT8 sobre el **mismo `loader_test`**
(partición estratificada, ~78 imágenes, las 3 clases). Además el BLOQUE 2-BIS añade un
`assert` que comprueba que train/val/test son disjuntos y **conserva su salida** — que
es literalmente la lección que dejó el docente en la Fase 2.

---

### 🔴 Hallazgo 2 — La cuantización era un no-op, y el informe lo explicaba mal

**Severidad: alta.**

```python
modelo_int8 = torch.quantization.quantize_dynamic(
    modelo_cpu, qconfig_spec={nn.Linear}, dtype=torch.qint8)
```

ResNet-18 tiene **una sola** capa `Linear`: la final, 512→3 = **1.539 parámetros de
11.178.051**, el **0,014%** del modelo. Todo lo demás son `Conv2d`, que la cuantización
dinámica **no toca**. Por eso el tamaño no bajó ni un byte (44,8 MB → 44,8 MB).

El informe atribuía el 0% a que *«los metadatos del modelo cuantizado pueden compensar
el ahorro en pesos»*. **Eso es incorrecto.** No se cuantizó nada.

Y el −17,8% de latencia, viniendo de 1.539 parámetros, no es atribuible a la
cuantización: es ruido de medición (FP32 se midió primero, con efectos de caché y sin
calentamiento previo).

**Corrección:**
- Se añade **cuantización estática PTQ** con `fbgemm`, fusión Conv+BN+ReLU y calibración
  sobre 128 imágenes de train. Esta sí alcanza las convoluciones.
- Se **conserva la dinámica como contraste documentado**, y el notebook imprime la
  fracción de parámetros cuantizables (0,014%) para que la explicación quede a la vista.
- La latencia se mide como **mediana de 30 corridas tras 10 de calentamiento**, en vez
  de media de 15 sin calentar.

> Convertir este error en una entrada de bitácora bien contada vale más que haberlo
> evitado: es una auditoría real sobre código propio, que es justo lo que pide el
> criterio 5.

---

### 🟠 Hallazgo 3 — El benchmark de tf.data no era comparable

**Severidad: media.** El informe reportaba **−85,4%** (el pipeline «optimizado» resultaba
casi el doble de lento) y lo explicaba como que el cache tarda en llenarse.

La explicación es incompleta. En el código, el pipeline **sin optimizar no tenía
`.shuffle()`** y el optimizado sí, con `buffer_size = len(rutas_imgs)` = 512 = el dataset
entero. Ese shuffle obliga a materializar las 512 imágenes antes de entregar el primer
batch.

No se estaba midiendo *cache vs no-cache*: se estaba midiendo *con barajado completo vs
sin barajado*. Es un artefacto de medición, no un comportamiento de TensorFlow.

**Corrección:** BLOQUE 2-BIS pone el **mismo `.shuffle()` en ambos** pipelines (solo
varían `num_parallel_calls`, `.cache()` y `.prefetch()`) y mide **época 1 y época 2 por
separado**, reportando la época 2 como la comparación válida.

---

### 🟠 Hallazgo 4 — La bitácora afirma haber usado la ruta estable; el log dice lo contrario

**Severidad: media, pero delicada** — es una afirmación de la bitácora que el propio
notebook desmiente, y la bitácora es el criterio donde la honestidad es la nota.

La bitácora registra como caso de auditoría el rechazo de `dynamo=True` y afirma haber
usado «la ruta estable». Pero la salida del notebook muestra:

```
[torch.onnx] Obtain model graph for `ResNet([...]` with `torch.export.export(..., strict=False)`...
```

`torch.export.export` **es** dynamo. En PyTorch 2.11 el exportador dynamo es el
**predeterminado**: no pasar `dynamo=True` no equivale a usar la ruta legacy. Hay que
pasar `dynamo=False` explícitamente.

**Corrección:** BLOQUE 8-BIS exporta con `dynamo=False` explícito (con `try/except` por
compatibilidad de versión) e imprime cuál ruta se usó realmente. La entrada de bitácora
se reescribe para contar lo que de verdad pasó — que es un caso de auditoría **mejor**
que el original, porque es un hallazgo propio y no el ejemplo que sugiere la guía.

---

### 🟡 Hallazgo 5 — El FP32 nunca se midió

**Severidad: media.** Bloquea el nivel Estratégico del criterio 2, que pide el ahorro
«documentado».

```python
print(f'VRAM estimada FP32:   {max(vram_fp16_list)*1.8:.0f} MB (aprox ×1.8)')
print(f'Tiempo/época FP32 est:{np.mean(tiempo_lista)*1.4:.1f}s (aprox ×1.4)')
```

Los «1084 MB» y «8,5 s» del informe son multiplicaciones de los valores FP16, no
mediciones. La bitácora lo declara honestamente («estimaciones ilustrativas basadas en
literatura»), lo cual está bien — pero **una época tarda 6 segundos**. Medirlo de verdad
cuesta menos de un minuto de cómputo.

**Corrección:** BLOQUE 4-BIS entrena 5 épocas en FP32 y 5 en FP16 con la misma semilla y
registra VRAM pico y segundos por época reales.

---

### 🟡 Hallazgo 6 — El modelo `.keras` se guardaba sin entrenar

**Severidad: media.** El código creaba `ResNet50V2` con pesos ImageNet y una cabeza
`Dense` **inicializada al azar**, la compilaba y la guardaba sin llamar a `fit()`. El
archivo de 95 MB no clasificaba agua: era un ResNet50V2 genérico con una capa aleatoria.

Como la ruta de despliegue declara que el `.tflite` sale de ese `.keras`, se estaría
desplegando un modelo sin entrenar.

**Corrección:** BLOQUE 8-BIS entrena la implementación Keras 5 épocas sobre la misma
partición y reporta su accuracy en test por separado, dejando explícito que `.pth` y
`.keras` son **dos implementaciones equivalentes**, no el mismo objeto.

---

### 🟡 Hallazgo 7 — El `.onnx` no carga si se entrega solo

**Severidad: baja, pero rompe la entrega.** El archivo `.onnx` pesa 90 KB porque los
pesos van en un sidecar `Centinela_Fase3_ResNet18.onnx.data` de 44,7 MB. El informe
reporta «44,8 MB», que es la suma — correcto como cifra, pero **si el `.onnx` viaja solo
al zip, no carga**.

**Corrección:** BLOQUE 8-BIS fuerza `save_as_external_data=False` y elimina el sidecar.

---

### 🟡 Hallazgo 8 — Faltaban requisitos explícitos de la guía

| Requisito de la guía | Estado original | Corrección |
|---|---|---|
| Comparar accuracy de validación con y sin augmentation (10 épocas) | ❌ No se hizo | BLOQUE 3-BIS |
| Resolver un CUDA OOM con evidencia (Estratégico, criterio 2) | ❌ No se hizo | BLOQUE 4-TER |
| Mixup o CutMix como técnica avanzada | ❌ No se hizo | Opcional — no bloquea nivel |
| Informe en **PDF, máx. 4 páginas** | ❌ `.docx`, ~7 páginas | Pendiente de reescritura |
| `README_Fase3.md` | ❌ Ausente (fase1 y fase2 sí lo tienen) | ✅ Creado |

---

### ⚪ Hallazgo 9 — Detalles de forma

- El informe dice **2026** en el encabezado y **2025** en el pie.
- Las figuras `.png` generadas no están embebidas en el documento.
- La guía especifica **modalidad individual** y nombre de archivo
  `Apellido_Nombre_Centinela_Fase3.zip`. Los tres documentos van firmados por Chiquiza ·
  Lerma · Villarraga. **Confirmar con el docente antes de entregar.**

---

## Parte C · Orden de ejecución sugerido

1. Correr `celdas_corregidas_fase3.py` en Colab, bloque por bloque, sobre el runtime ya
   inicializado hasta el BLOQUE 1.
2. Recoger `cifras_informe.json`.
3. Reescribir el informe con esas cifras — **ninguna a mano**. Máximo 4 páginas, a PDF.
4. Reescribir las entradas de bitácora afectadas (ver Parte D).
5. Reempaquetar con la estructura del README.

---

## Parte D · Entradas de bitácora a reescribir

| Entrada actual | Problema | Cómo debe quedar |
|---|---|---|
| §3.4 «tamaño INT8 igual a FP32 → los metadatos compensan» | Explicación técnicamente falsa | Auditoría real: la cuantización dinámica solo alcanza `Linear`; en ResNet-18 eso es el 0,014% del modelo. Se corrigió con PTQ estática |
| §5 «caso de auditoría: `dynamo=True`» | El export sí pasó por dynamo; y el caso lo sugiere la guía | Auditoría propia: descubrimos por el log que dynamo es el default en 2.11 y que hay que pedir `dynamo=False` explícitamente |
| §3.1 «cache más lento en la primera medición» | Explicación incompleta | El causante era el `.shuffle()` presente solo en el pipeline optimizado; se corrigió el diseño del benchmark |
| §3.3 «estimación FP32 como FP16 × 1,8» | Honesta pero evitable | Se reemplaza por medición directa |
| — (nueva) | — | **Entrada nueva:** detección del accuracy INT8 medido sobre 128 imágenes de una sola clase y del conjunto de entrenamiento. Es el hallazgo de mayor valor de la fase |

---

## Parte E · Impacto estimado en la rúbrica

| Criterio | Peso | Antes | Después | Qué lo mueve |
|---|---|---|---|---|
| 1 · Pipeline + Augmentation | 20% | ~4,2 | ~4,7 | Benchmark justo + A/B de augmentation |
| 2 · Entrenamiento GPU | 20% | ~4,3 | ~4,8 | FP32 medido + CUDA OOM documentado |
| 3 · Optimización Edge | 20% | ~4,0 | ~4,7 | Cuantización real + accuracy sobre el test correcto |
| 4 · Ruta de despliegue | 20% | ~4,5 | ~4,6 | Ya era lo más fuerte |
| 5 · Uso responsable de IA | 10% | ~4,2 | ~4,8 | Auditorías propias en vez de la sugerida por la guía |
| 6 · Comunicación | 10% | ~4,2 | ~4,6 | PDF 4 páginas + README + cifras trazables |

**Estimado: 4,2 → ~4,7 (nivel Estratégico).**

Es orientativo: la calificación la asigna el docente y la sustentación pesa aparte.

---

## Parte F · Preparación para la sustentación

Las tres preguntas que el docente anunció explícitamente:

**1. Sobre la clave de alineación.** *«¿Qué valor sale de `cond_train.mean()*100`?»*
→ Sale **100,0%**. La clave no varía, la clase Limpia nunca se usó y la imagen entró
como ruido. La tesis de «redundancia» del informe de Fase 2 **no se sostiene**: no era
redundante, no había información con la cual serlo. La causa fue un error de unidades
(multiplicar por 10 rasgos ya normalizados a [0,1] y compararlos contra umbrales de pH
físicos).

**2. Sobre la ganancia de la fusión.** *«Mismo ROC-AUC pero más precisión y menos recall,
¿qué significa geométricamente?»*
→ Es un **desplazamiento sobre la misma curva ROC**: mismo ordenamiento leído en otro
umbral, no información nueva. Y contradice nuestra propia recomendación de priorizar el
recall, porque lo baja de 0,879 a 0,825.

**3. Sobre el horizonte de predicción.** *«La ventana abarca [i−14, i−1] y la etiqueta es
el pH del día i+1. ¿Un día o dos?»*
→ Son **dos días** de anticipación; el día `i` no lo usa nadie. El error va en dirección
conservadora (la tarea real es más difícil que la declarada), pero la promesa al cliente
debe precisarse.

**Cuarta pregunta probable, sobre esta fase:** *«Su tabla INT8 dice que el accuracy sube
de 97,4% a 100%. ¿Cómo se explica?»*
→ **No se explica: era un error de medición y lo detectamos nosotros.** Se evaluaba
sobre 128 imágenes de una sola clase y del conjunto de entrenamiento. Corregido y
documentado en la bitácora.

---

*Universidad Santo Tomás · Maestría en Ciencia de Datos · 2026*
