# Revisión profunda del pipeline de segmentación

Auditoría técnica completa del segmentador, contrastada con literatura y práctica
actual en detección de enfermedades foliares. Fecha: 2026-09-08. Commit base:
`985a920`.

El documento no cambia código, datasets ni locks. Cada hallazgo indica archivo,
línea, evidencia reproducible e impacto. Los hallazgos se ordenan por severidad,
no por etapa del pipeline.

## Resumen ejecutivo

El proyecto está por encima de la práctica habitual en trazabilidad: locks
SHA-256, splits agrupados con dedupe perceptual, piloto retenido, gates de
confirmación explícita y métricas orientadas a recall de píxel en lugar de mAP
puro. Eso resuelve correctamente los problemas que más suelen invalidar trabajos
de segmentación foliar.

Los problemas reales no están en la trazabilidad sino en tres puntos:

1. **La evaluación final de `test` no mide el pipeline de producción.** Usa
   `model.val(save_txt=True)`, que en Ultralytics guarda predicciones con
   `conf=0.001` y todas las instancias, mientras producción usa umbral de
   selección `0.50` y una sola hoja seleccionada. Las métricas downstream de
   `test` no son comparables con las de D-01.
2. **El quality gate calibrado colapsa en un único estadístico frágil y
   dependiente de resolución.** `normalized_perimeter` crece con la resolución
   para bordes reales (verificado empíricamente abajo), y el dataset mezcla
   224×224 con 3048×4064. El gate penaliza justamente las hojas de alta
   resolución con borde dañado, que son las diagnósticamente críticas.
3. **La ruta de evaluación de `test` está clavada al checkpoint baseline**, no
   al candidato D-01, así que el paso 4 del protocolo de mejora no se puede
   ejecutar sin tocar código.

Además hay deriva entre documentación, configuración y valores por defecto del
código en los thresholds del gate, y el consumidor externo del gate recibe por
defecto los umbrales *previos* a la calibración.

## Mapa del pipeline

| Etapa | Módulos | Estado |
|---|---|---|
| Consolidación de fuentes | `src/data/segmentation_consolidation.py`, `segmentation_audit.py`, `jpeg_normalization.py` | Congelado, lock `7a4a5c08` |
| Revisión humana y finalización | `segmentation_review.py`, `segmentation_review_preview.py`, `segmentation_finalization.py` | Congelado |
| Splits agrupados | `segmentation_split.py` | Congelado, semilla 42, 809/173/173 |
| Preflight | `src/training/segmentation_preflight.py` | Verifica locks + smoke loader |
| Empaquetado y entrenamiento | `scripts/package/`, `cloud_training/`, `modal_training.py` | D-01 completado en Modal |
| Inferencia | `src/segmentation/leaf_segmenter.py` | Wrapper Ultralytics con versión fijada |
| Selección de hoja | `src/preprocessing/segmented_leaf_processor.py` | Score `0.45·área + 0.35·centro + 0.20·conf` |
| Máscara y salida | `src/preprocessing/leaf_mask.py`, `letterbox.py` | 4 perfiles, fondo RGB exacto |
| Quality gate | `src/segmentation/quality.py` | v1.0.0, calibrado sobre 42 imágenes |
| Métricas downstream | `src/evaluation/segmentation_downstream.py` | Recall de píxel priorizado |

---

## Hallazgos de severidad alta

### H-01 — La evaluación de `test` mide un pipeline distinto al de producción

**Evidencia.** `cloud_training/run_ultralytics.py:988-1041`: la evaluación final
llama `model.val(**config, plots=True, save_json=True, save_txt=True,
save_conf=False)` y después alimenta `evaluate_downstream()` con
`<save_dir>/labels`. `cloud_training/configs/validate_yolo26n_seg.yaml` no fija
`conf`.

En Ultralytics el default de `conf` es **0.001 en modo `val`** y 0.25 en
`predict` ([Ultralytics
cfg](https://docs.ultralytics.com/usage/cfg/)). Es decir, las etiquetas que
alimentan `downstream_per_image.csv` contienen **todas** las propuestas por
encima de 0.001 y **todas** las instancias, mientras que el pipeline real:

- descarta propuestas bajo `selection_confidence_threshold: 0.50`;
- selecciona **una sola** instancia (`select_target_leaf`);
- aplica el quality gate y puede caer en fallback.

**Impacto.** Tres métricas del resumen quedan sin significado en `test`:

- `leaf_pixel_recall` queda inflado: la unión de decenas de propuestas ruidosas
  cubre casi toda la hoja por construcción;
- `over_segmentation_ratio` queda inflado en el sentido contrario;
- `fallback_rate` e `images_without_detection_rate` son ≈0 estructuralmente,
  porque con `conf=0.001` casi nunca falta detección.

La evaluación D-01 sobre las 150 imágenes de enfermedades sí usó el wrapper real
(`docs/es/leaf-detection/segmentation-d01-results.md`), así que existen dos
métricas con el mismo nombre y semántica incompatible. Comparar el `test`
congelado contra el número de D-01 (IoU 0.98122) sería un error de lectura.

**Recomendación.** Separar explícitamente:

- `val_metrics`: `model.val()` para mAP/AP, sin `save_txt` ni downstream;
- `pipeline_metrics`: recorrer `test` con `UltralyticsLeafSegmenter` +
  `SegmentedLeafProcessor` + `assess_segmentation`, exactamente como hace
  `scripts/experiments/calibrate_segmentation_selection_threshold.py`, y calcular
  ahí el downstream. Ese script ya contiene el 90 % del código necesario;
  conviene extraer un `src/evaluation/pipeline_evaluation.py` reutilizable.

Mientras no se separe, marcar `downstream_summary` de `evaluate_mode` como
diagnóstico de la cabeza del modelo, nunca como métrica del producto.

### H-02 — El gate de `test` está clavado al checkpoint baseline, no al candidato

**Evidencia.** `cloud_training/run_ultralytics.py:51-53` y `465-477`:

```python
EXPECTED_BEST_CHECKPOINT_SHA256 = "4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f"
...
expected_checkpoint = (OUTPUTS / "segmenter/yolo26n_seg_baseline/weights/best.pt").resolve()
```

El candidato actual documentado es
`outputs/leaf_detection/segmenter/d01_mosaic0_seed42/weights/best.pt` con SHA-256
`a2bf4f20...`. Cualquier intento de correr `make leaf-segmentation-cloud-test`
sobre D-01 aborta con `Checkpoint de evaluación inesperado`.

**Impacto.** El paso 4 del protocolo de mejora ("ejecutar `test` una sola vez con
configuración y checkpoint congelados") está bloqueado por código. Peor: la
tentación natural es editar la constante justo antes de correr el test, lo que
elimina la garantía de "una sola vez" que la constante pretendía dar.

**Recomendación.** Convertir el pin de checkpoint en un artefacto de datos, no en
una constante de código: un `outputs/leaf_detection/segmenter/promoted_checkpoint.json`
escrito con `open(..., "x")` (mismo patrón que
`register_verified_experiment_weights.py`), con `path`, `sha256`, `experiment_id`
y `promoted_at`. `evaluate_mode` lee ese archivo y verifica contra él. Así el pin
sigue siendo inmutable y auditable, pero no requiere tocar el runner para
promover un experimento.

### H-03 — `min_multi_instance_score_margin` y el default de `SegmentationQualityGateConfig` no coinciden con la calibración

**Evidencia.** Tres fuentes de verdad divergentes para el mismo gate:

| Threshold | `config/segmentation.yaml` | Default en `quality.py:50-56` | Doc `segmentation-reliability-gate-audit.md` |
|---|---:|---:|---:|
| `large_mask_area_ratio` | 0.25 | 0.50 | 0.50 |
| `min_large_mask_bbox_ratio` | 0.80 | 0.70 | 0.70 |

Además `assess_segmentation(result)` acepta `quality_gate=None` y en ese caso usa
`SegmentationQualityGateConfig()`, es decir los valores **previos** a la
calibración (`src/segmentation/quality.py:265`). El único consumidor que pasa la
configuración real es `scripts/experiments/audit_segmentation_reliability.py:281`.

**Impacto.** El proyecto del clasificador —el consumidor real de esta librería—
que llame `assess_segmentation(result)` sin argumentos obtiene silenciosamente el
gate no calibrado. Y `reject_multiple_eligible` tiene default `True` en la
función pero `false` en el YAML, lo que invierte la política multi-hoja.

**Recomendación.**

1. Alinear los defaults del dataclass con los valores calibrados, o quitar los
   defaults y exigir configuración explícita (preferible: falla ruidosa).
2. Añadir `src/segmentation/config.py` con un único
   `load_segmentation_config(path) -> SegmentationRuntimeConfig` que devuelva
   `LeafMaskProcessorConfig`, `SegmentationQualityGateConfig`, `reject_multiple_eligible`,
   ruta de checkpoint y SHA esperado. Hoy cada script reparsea el YAML por su
   cuenta.
3. Actualizar el bloque YAML de `segmentation-reliability-gate-audit.md`, que
   está desactualizado respecto a la configuración activa.

### H-04 — `normalized_perimeter` no es invariante a resolución sobre bordes reales

**Evidencia.** `src/preprocessing/leaf_mask.py:229-231` define el perímetro como
conteo de aristas de rejilla y `normalized_perimeter = perímetro / sqrt(área)`.
Para un borde liso eso es invariante a escala; para un borde con textura real no
lo es. Reproducción:

```python
# rasteriza la misma forma a varias resoluciones, con y sin rugosidad de borde
python - <<'PY'
import math, numpy as np
from PIL import Image, ImageDraw
from src.preprocessing.leaf_mask import mask_geometry

def leaf(n, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    img = Image.new("L", (n, n), 0); d = ImageDraw.Draw(img); pts = []
    for t in np.linspace(0, 2*math.pi, 400, endpoint=False):
        r = 0.42*(1+0.35*math.cos(3*t))
        if jitter: r *= (1 + jitter*rng.normal())
        pts.append(((0.5+r*math.cos(t))*n, (0.5+0.55*r*math.sin(t))*n))
    d.polygon(pts, fill=255); return img

for n in (224, 640, 1280, 3048):
    print(n, round(mask_geometry(leaf(n)).normalized_perimeter, 3),
             round(mask_geometry(leaf(n, jitter=0.02, seed=1)).normalized_perimeter, 3))
PY
```

Resultado medido:

| Resolución | Borde liso | Borde rugoso (2 %) |
|---:|---:|---:|
| 224 | 5.488 | 7.458 |
| 640 | 5.520 | 8.155 |
| 1280 | 5.514 | 8.245 |
| 3048 | 5.509 | 8.291 |

La misma forma cruza el umbral `max_large_mask_normalized_perimeter = 8.0`
únicamente por resolución.

**Impacto.** El dataset mezcla `corn` (155 imágenes a 224×224) con
`corn_leaf_diseases_classification` (hasta 3048×4064). El gate es
sistemáticamente más estricto con las imágenes grandes. Y el borde rugoso es
exactamente la firma visual del daño por cogollero, tizón avanzado y hojas
partidas: **el gate está sesgado contra los casos que más importa diagnosticar**.
El umbral 8.0 se calibró sobre una muestra de 42 imágenes cuya mezcla de
resoluciones no se controló.

**Recomendación.**

- Normalizar la medición: rasterizar la máscara a un canvas de referencia fijo
  (p. ej. 640 en el lado mayor, con `NEAREST`) antes de medir perímetro, o
  aplicar un suavizado morfológico de radio proporcional a `sqrt(área)`.
- Alternativa mejor fundamentada: sustituir `normalized_perimeter` por
  **Boundary IoU** ([Cheng et al., CVPR
  2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf)),
  que es la métrica estándar para calidad de contorno y sí está diseñada para ser
  balanceada entre escalas. Requiere ground truth, así que sirve para evaluación,
  no para el gate en línea; para el gate en línea, la versión normalizada por
  canvas es suficiente.
- Registrar `resolution_bin` como columna de la auditoría del gate y reportar la
  precisión del gate desagregada por bin.

### H-05 — El gate geométrico colapsa a un solo criterio en la práctica

**Evidencia.** `src/segmentation/quality.py:299-306`: la regla
`suspicious_large_mask_geometry` exige las tres condiciones a la vez
(`área ≥ 0.25` **y** `mask_bbox_ratio < 0.80` **y** `normalized_perimeter > 8.0`).

Para una hoja de maíz —alargada y casi siempre diagonal— `mask_bbox_ratio` es
estructuralmente bajo. Medición sobre una hoja diagonal sintética típica:

```
area_ratio=0.123  mask_bbox_ratio=0.166  normalized_perimeter=9.81
```

`mask_bbox_ratio < 0.80` se cumple prácticamente siempre para una hoja real, y
`área ≥ 0.25` se cumple en la mayoría del dataset (hojas a cuadro casi completo).
Las dos primeras condiciones no discriminan: el gate depende de facto sólo de
`normalized_perimeter`, es decir del estadístico frágil de H-04.

**Impacto.** La combinación H-04 + H-05 significa que el gate tiene un único
punto de falla y está sesgado por resolución. La calibración reporta 24/24
máscaras `GOOD` conservadas y 0 falsos confiables, pero eso es sobre la muestra
en la que se calibró (el propio documento lo reconoce).

**Recomendación.** Añadir señales ortogonales que sí discriminen mal-segmentación
de hoja diagonal legítima, todas ya calculadas en `MaskGeometry`:

- `largest_component_ratio < 0.95` → máscara fragmentada (unión de dos hojas o
  fuga de fondo);
- `connected_components > 1`;
- `border_contact_count == 4` combinado con `mask_area_ratio` alto → fuga a fondo;
- `bbox_aspect_ratio` fuera de un rango plausible para hoja de maíz.

Estas variables ya se persisten en `AUDIT_NUMERIC_FIELDS`, así que se pueden
evaluar retroactivamente sobre la auditoría existente sin nueva anotación.

### H-06 — Riesgo de duplicados por inferencia end-to-end de YOLO26 no mitigado

**Evidencia.** `src/segmentation/leaf_segmenter.py:264-274` pasa `iou`,
`max_det` y `agnostic_nms` a `model.predict()`. YOLO26 es **NMS-free por
defecto**: la cabeza usa asignación uno-a-uno y produce detecciones finales sin
paso de supresión ([Ultralytics
YOLO26](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment)).
Existe un reporte abierto de **detecciones duplicadas con ~100 % de solape
específicamente en modelos `yolo26*-seg` en modo end2end**
([ultralytics#23685](https://github.com/ultralytics/ultralytics/issues/23685)),
donde el workaround del reportante fue `end2end=False, nms=True`.

**Impacto.** Si el modelo corre en modo end2end, `iou_threshold: 0.70` y
`max_detections: 20` son parámetros muertos: se registran en `to_metadata()` como
si estuvieran aplicándose, y `"nms": "class_aware"` en el metadato sería falso.
Peor, dos duplicados casi idénticos producen `eligible_instances = 2` con
`instance_score_margin ≈ 0`, que dispara `ambiguous_instance_score_margin` y
convierte un `reliable` correcto en `uncertain`. Es un modo de fallo silencioso
que degrada cobertura sin degradar ninguna métrica de máscara.

**Recomendación.**

1. Verificar en el checkpoint entrenado si `model.model.end2end` es `True` y
   registrarlo en `to_metadata()` en lugar de la cadena fija `"class_aware"`.
2. Añadir dedupe de instancias en el adaptador antes de `select_target_leaf`:
   descartar la instancia de menor confianza de cualquier par con mask-IoU por
   encima de ~0.9. Son pocas instancias (`max_det=20`), el coste es
   despreciable y elimina la dependencia del comportamiento interno de
   Ultralytics.
3. Añadir un test con un `LeafSegmenter` falso que devuelva dos máscaras
   idénticas y afirmar que el resultado sigue siendo `reliable`.

---

## Hallazgos de severidad media

### H-07 — `crop_mask_letterbox` reintroduce el halo que el resto del pipeline evita

**Evidencia.** `src/preprocessing/segmented_leaf_processor.py:598-603` llama
`letterbox_image(..., resample=BILINEAR)` (default en `letterbox.py:72`) sobre un
recorte ya enmascarado a negro exacto. La interpolación bilineal mezcla el borde
de la hoja con el fondo:

```
letterbox: 173 colores únicos, 729 píxeles que no son ni negro puro ni color de hoja
```

Esto contradice explícitamente el resto del diseño: `apply_leaf_mask` garantiza
RGB(0,0,0) exacto y `save_debug_artifacts` usa PNG precisamente "porque JPEG
introduciría ringing distinto de cero"
(`segmented_leaf_processor.py:465`).

**Impacto.** El perfil `crop_mask_letterbox` entrega al clasificador un halo
verde-a-negro de ~1 px alrededor de toda la hoja. Un clasificador puede aprender
ese contorno como atajo (`shortcut learning`), que es exactamente el riesgo que
la máscara pretendía eliminar.

**Recomendación.** Redimensionar imagen y máscara por separado (`BILINEAR` para
la imagen, `NEAREST` para la máscara) y reaplicar `apply_leaf_mask` después del
resize. `LetterboxResult` ya expone `scale` y `padding`, así que la máscara se
puede transformar con la misma geometría.

### H-08 — `rasterize_polygons` deforma a cuadrado y usa `size - 1`

**Evidencia.** `src/evaluation/segmentation_downstream.py:100-111` rasteriza
siempre a `size × size` (640×640) y mapea coordenadas normalizadas con
`x * (size - 1)`.

**Impacto.**

- La deformación a cuadrado es teóricamente inocua para IoU (el escalado
  anisótropo preserva razones de área), pero degrada la discretización de
  estructuras finas en imágenes muy apaisadas (900×600, 4064×3048). Es una
  fuente de ruido que crece con la relación de aspecto y no se reporta.
- `size - 1` en vez de `size` encoge sistemáticamente ambas máscaras ~0.3 %. Se
  cancela en IoU y Dice, pero sesga `truth_area_fraction` y
  `predicted_area_fraction`, que sí se usan para el binning `_area_bin`.

**Recomendación.** Rasterizar preservando la relación de aspecto (lado mayor =
`raster_size`) y usar `x * size` con `ImageDraw` en coordenadas de píxel. Añadir
`raster_aspect_error` al resumen para hacer visible el residuo de discretización.

### H-09 — El selector puede elegir la hoja equivocada en escenas multi-hoja

**Evidencia.** `select_target_leaf` puntúa
`0.45·área_relativa + 0.35·proximidad_al_centro + 0.20·confianza`. No hay ninguna
señal relacionada con síntomas.

**Impacto.** En una foto de campo con varias hojas, la hoja más grande y centrada
no es necesariamente la lesionada. El manifiesto de auditoría documenta este caso
("La propuesta cubre la hoja diagnóstica principal") pero sobre sólo 42 imágenes,
de las que una fracción es multi-hoja. La política actual es defendible —el
usuario apunta la cámara a la hoja que le preocupa— pero es una **suposición no
verificada** que debe quedar escrita como tal.

**Recomendación.**

- Documentar la suposición de encuadre en el contrato de la API, no sólo en el
  código.
- Instrumentar: cuando `eligible_instances > 1` y `reject_multiple_eligible` es
  `false`, persistir siempre las máscaras de las instancias descartadas en el
  bundle de debug, para poder auditar decisiones a posteriori.
- A medio plazo, cuando exista el dataset de casos difíciles (≥30 multi-hoja
  según el protocolo), medir directamente la tasa de acierto del selector contra
  la hoja marcada por el anotador, y considerar devolver las N mejores instancias
  para que el clasificador agregue en lugar de decidir aquí.

### H-10 — Aumentos de datos por defecto inadecuados para hojas

**Evidencia.** Ningún perfil en `cloud_training/configs/` fija `degrees`,
`flipud`, `mask_ratio`, `overlap_mask`, `scale`, `hsv_*` ni `close_mosaic`.
Se usan los defaults de Ultralytics: `degrees=0.0`, `flipud=0.0`, `fliplr=0.5`,
`mask_ratio=4`, `overlap_mask=True`.

**Impacto.**

- **`degrees=0.0` y `flipud=0.0`.** Una hoja de maíz no tiene orientación
  canónica y una foto de campo se toma en cualquier ángulo. El proyecto ya
  registra `orientation` como variable de estratificación, señal de que sabe que
  importa. Sin rotación ni volteo vertical, el modelo hereda el prior de
  orientación del corpus de laboratorio.
- **`mask_ratio=4`.** Las máscaras de ground truth se rasterizan a `imgsz/4`
  = 160×160 para la pérdida. Dado que la métrica prioritaria del proyecto es
  fidelidad de contorno (recall de píxel, sub-segmentación), es la configuración
  menos favorable posible. `mask_ratio=1` es la ablación obvia que falta.

**Recomendación.** Añadir dos ablaciones al plan, ambas de coste ≈25 min de A10
según el registro de D-01:

- `D-07`: `mask_ratio: 1` sobre la configuración D-01.
- `D-08`: `degrees: 20`, `flipud: 0.5` sobre la configuración D-01.

Ambas son cambios de una sola variable, compatibles con el protocolo existente y
con la regla de no usar `test` para seleccionar.

### H-11 — La calibración del gate se evalúa sobre la misma muestra en que se ajusta

**Evidencia.** `src/evaluation/segmentation_gate_calibration.py:148-160`:
`candidate_gates()` enumera 2×4×4×4×5 = **640 configuraciones**, todas evaluadas
sobre las **42 filas** de `segmentation_reliability_audit_v1.csv`, y
`calibrate_gate` devuelve la mejor sobre esas mismas filas.

**Impacto.** 640 candidatos contra 42 puntos, con 24 positivos, es un régimen de
sobreajuste por comparaciones múltiples. "Conserva 24/24 GOOD y reduce falsos
confiables de 1 a 0" es una métrica de entrenamiento, no una estimación de
generalización. El intervalo de confianza binomial al 95 % para "0 falsos
confiables de 18 no-GOOD" llega hasta ~18 %; la evidencia no distingue un gate
excelente de uno mediocre.

El protocolo de mejora ya lo reconoce ("es una estimación sobre la misma muestra
usada para calibrar"), lo cual es honesto. El problema es que el valor calibrado
ya está en producción.

**Recomendación.**

- Reportar, junto al gate recomendado, el intervalo de Wilson al 95 % de
  `reliability_precision` y `good_mask_coverage`.
- Reducir la rejilla o penalizar explícitamente por complejidad; hoy
  `_distance(candidate, baseline)` es sólo el último criterio de desempate.
- Priorizar la ronda de 300 imágenes difíciles del protocolo **como conjunto de
  validación del gate**, no sólo como datos de entrenamiento del modelo. Es la
  acción de mayor valor pendiente en todo el proyecto.

### H-12 — La identidad del checkpoint de inferencia no está fijada

**Evidencia.** `config/segmentation.yaml:7` apunta a
`leaf_detection/models/doctor_maiz_leaf_segmenter_best.pt`. Ningún script del
repositorio promueve, copia ni verifica ese archivo. `UltralyticsLeafSegmenter`
calcula `checkpoint_sha256` (`leaf_segmenter.py:229`) pero **nunca lo compara
contra un valor esperado**.

**Impacto.** Contraste llamativo: el runner de entrenamiento verifica el SHA-256
del checkpoint, del paquete, del dataset y hasta del entorno Python antes y
después de evaluar; la ruta de inferencia —la que llega al usuario final— acepta
cualquier `.pt` en esa ruta. Un checkpoint sustituido por accidente produce
resultados plausibles sin ninguna alarma.

**Recomendación.** Añadir `segmentation.checkpoint_sha256` al YAML y verificarlo
en `_load_model()`, con el mismo mensaje de error que la verificación de versión
de Ultralytics que ya existe justo al lado.

---

## Hallazgos de severidad baja

### H-13 — `to_metadata()` codifica el nombre del modelo en duro

`src/segmentation/leaf_segmenter.py:303` devuelve `"segmenter_model": "yolo26n-seg"`
como literal. Si se promueve D-04 (`yolo26s-seg`), todo el metadato de
procedencia quedará mintiendo. Derivarlo del checkpoint cargado o de la
configuración.

### H-14 — `src/training/segmentation_preflight.py` importa `torch` a nivel de módulo

`segmentation_preflight.py:20`. Ese módulo lo importa
`scripts/package/leaf_segmentation_make.py:14`, que a su vez sirve targets
puramente de sólo lectura (`status`, `verify-locks`, `verify-splits`,
`cloud-package-verify`). Consecuencia: una verificación de locks exige ~2 GB de
PyTorch. Contradice el patrón explícito de importación diferida que sí aplica
`leaf_segmenter.py` con Ultralytics. Mover `import torch` dentro de las dos
funciones que lo usan (`_environment_probe` y el smoke loader).

### H-15 — `make check` no corre `ruff` y `ruff format` no se verifica en ninguna parte

`Makefile:352-362`: `lint` corre `ruff check src/ scripts/`, `check` corre sólo
`pyright`. `CLAUDE.md` lista `make lint` y `make check` como verificación. Ni
`tests/` está cubierto por lint, ni el formato se verifica: `ruff format --check .`
reporta **47 archivos** que se reformatearían. Recomendación: `check: lint fmt-check`
y añadir `tests/` a los paths.

### H-16 — Dos tests no son herméticos

`python -m pytest` en un clon limpio: **297 pasan, 2 fallan**.

- `tests/checks/test_makefile_safety.py::test_status_is_read_only` requiere
  `data/leaf_detection/.../dataset_lock.json`, que está en `.gitignore`. Debería
  `skipUnless` el archivo exista.
- `tests/package/test_leaf_segmentation_make_helper.py::test_package_verify_rejects_bad_sha_before_extraction`
  requiere `torch` instalado (consecuencia de H-14). Se resuelve solo al arreglar
  H-14.

Además, tres módulos de test fallan en *colección* sin `torch`
(`test_leaf_segmentation_cloud_package.py`, `test_run_ultralytics_summary.py`,
`test_segmentation_preflight.py`), lo que interrumpe la suite completa en lugar de
saltarse esos módulos. Un `pytest.importorskip("torch")` al inicio de cada uno
mantiene la suite verde en un entorno base.

### H-17 — El contrato 183 vs 182 instancias sigue abierto y bloquea la evaluación

`run_ultralytics.py:47` fija `EXPECTED_TEST_INSTANCE_COUNT = 183` y el runner
aborta si Ultralytics reporta otra cifra. La discrepancia documentada (183
anotaciones raw frente a 182 instancias efectivas, punto 3 del protocolo de
mejora) hará fallar la evaluación final. Debe resolverse **antes** de gastar la
única ejecución de `test`: identificar el polígono degenerado concreto,
documentarlo y decidir si el contrato son 182 o 183.

### H-18 — La proximidad al centro se mide sobre el centroide, no sobre el bbox

`_center_proximity` (`segmented_leaf_processor.py:277-287`) usa el centroide de la
máscara. Para una hoja en forma de "L" o muy curvada, el centroide puede caer
fuera de la hoja y desplazarse mucho, lo que hace el score sensible a la forma y
no sólo a la posición. Usar el centro del bbox, o reportar ambos, haría el score
más estable. Impacto bajo dado el peso de 0.35 y que hoy casi siempre hay una
sola instancia elegible.

---

## Lo que está bien resuelto

Vale la pena registrarlo porque son decisiones que la literatura reciente
identifica como fallos habituales y aquí ya están cubiertas:

- **Splits agrupados con dedupe perceptual y variantes Roboflow.** La fuga por
  near-duplicates es el error metodológico más común en datasets de enfermedades
  foliares, y aquí se controla con componentes conexos sobre SHA-256, nombre base
  y hash perceptual con distancia Hamming ≤ 4.
- **Piloto externo retenido y nunca usado para seleccionar.** Es la única defensa
  real contra el `reliability gap` documentado en transferencia
  laboratorio→campo.
- **Métricas orientadas a recall de píxel en lugar de mAP.** La justificación en
  el docstring de `segmentation_downstream.py` (recortar tejido destruye señal
  irrecuperable; sobre-segmentar sólo añade fondo) es correcta y poco frecuente.
  Es exactamente el razonamiento adecuado para un segmentador que alimenta un
  clasificador.
- **Desagregación por fuente, resolución, orientación, contacto con borde y
  número de hojas.** Hace visible la brecha de dominio antes de que se convierta
  en un fallo de campo.
- **Gates de confirmación literal separados para smoke, train y resume**, y
  escritura de artefactos con `open(..., "x")` para no sobrescribir evidencia.
- **`binary_mask_array` rechaza valores intermedios.** Detecta máscaras
  redimensionadas por interpolación bilineal, que es una fuente clásica de
  geometría corrupta silenciosa.

---

## Investigación externa

### Segmentación como preprocesamiento de clasificación foliar

La evidencia apoya la decisión de fondo del proyecto. Trabajos con DeepLabV3+ y
CNN de clasificación reportan mejoras al eliminar fondo antes de clasificar, y
revisiones sobre segmentación semántica para clasificación de enfermedades
foliares coinciden en que eliminar información engañosa aumenta la exactitud y
reduce tiempo de procesamiento
([Frontiers in Plant Science, 2022](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1031748/full);
[ScienceDirect, 2024](https://www.sciencedirect.com/science/article/pii/S277237552400131X)).

Hay un matiz importante que conviene incorporar: el enmascarado **no siempre**
ayuda, y depende de la arquitectura del clasificador. Trabajos sobre estrategias
de enmascarado distinguen *early masking* (enmascarar píxeles antes del
clasificador, que es lo que hace `mask_black`) de *late masking* (enmascarar en
el mapa de características antes del pooling global), y reportan que en varias
arquitecturas el enmascarado temprano degrada resultados respecto a la imagen
completa
([Masking Strategies for Background Bias Removal, 2023](https://arxiv.org/abs/2308.12127);
[impacto del background removal en clasificación, 2023](https://arxiv.org/html/2308.09764v2)).

**Implicación para este proyecto.** El perfil de salida no debería fijarse aquí
por decreto. `config/segmentation.yaml` ya soporta cuatro perfiles
(`mask_black`, `bbox_crop`, `crop_mask_black`, `crop_mask_letterbox`); el
proyecto del clasificador debería medir los cuatro más el baseline sin
segmentación, y el segmentador debería exponer la máscara como canal separado
para permitir *late masking* sin re-ejecutar inferencia. Un quinto perfil
`mask_alpha` (RGB original + máscara en canal A) cubriría ese caso con un cambio
mínimo.

### Brecha laboratorio→campo

Es el riesgo dominante y está bien caracterizado. Mohanty et al. (2016) midieron
caída a ~31 % al evaluar en imágenes de campo modelos entrenados en PlantVillage.
Para maíz específicamente, un modelo con 94.1 % en el test de PlantVillage para
gray leaf spot cayó a 55.1 % en imágenes de campo
([PMC9330607](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330607/)). Trabajo
reciente cuantifica el fenómeno como un fallo simultáneo de exactitud,
calibración de confianza, predicción selectiva y rechazo de clases desconocidas,
y encuentra **eficacia limitada de las técnicas estándar de mitigación**
([PMC13236948, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13236948/)). El
sesgo de fondo de PlantVillage está documentado explícitamente
([arXiv:2206.04374](https://arxiv.org/abs/2206.04374)).

**Implicación.** Las métricas de D-01 (IoU 0.98, Dice 0.99) están medidas sobre
`val` del mismo corpus consolidado, en su mayoría condiciones controladas. No son
predictivas de campo. El único número informativo hoy es el peor caso: 11.41 % de
tejido recortado en una imagen 900×600 con fondo complejo. La ronda de 300
imágenes difíciles del protocolo es la prioridad correcta, y su composición
(cogollero severo, oclusión, hoja parcial en borde, hoja pequeña, fondo complejo,
multi-hoja) coincide con lo que la literatura identifica como los ejes de la
brecha.

### Anotación asistida para la ronda de datos difíciles

SAM/SAM2 se ha consolidado como acelerador de anotación en agricultura: ARAMSAM
orquesta SAM 1 y SAM 2 para pre-etiquetado sobre datasets agrícolas
([Frontiers in AI, 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1748468/full)),
y existen pipelines SAM→YOLOv8 para hojas
([Agronomy 15(5):1081](https://doi.org/10.3390/agronomy15051081)). Nota
importante: SAM 2 no supera consistentemente a SAM 1 y depende fuertemente del
ajuste de sus generadores automáticos de máscara.

**Implicación.** El protocolo actual exige revisión humana y prohíbe máscaras
generadas como verdad de referencia. Esa regla es correcta y no debe relajarse.
Pero SAM es compatible con ella si se usa como **pre-etiquetado que un humano
corrige y aprueba**, con el estado de anotación registrado en
`hard_example_intake_template.csv` (que ya tiene columnas `annotation_status`,
`annotator`, `reviewer`). Conviene añadir una columna `prelabel_source` para
distinguir "anotado desde cero" de "SAM corregido", y medir si esa distinción
correlaciona con calidad.

### Fuentes de datos de campo para maíz

Para la ronda difícil existen corpus públicos relevantes: MahindiNet (plagas y
enfermedades de maíz con foco en cogollero), el dataset de Namibia University
(MSV + FAW), el de KaraAgroAI en Ghana (FAW), y datasets de detección temprana de
plagas en hoja de maíz tomados con smartphone en campo a distintas horas y
localizaciones
([PMC11905841](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11905841/);
[Frontiers, 2026](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1803005/full)).

**Advertencia.** Casi ninguno trae máscaras de instancia; son clasificación o
detección. Servirían como fuente de imágenes para anotar, no como ground truth
de segmentación, y requieren verificar licencia. Eso encaja con el proceso ya
definido en el protocolo (procedencia/licencia obligatoria por imagen).

### Estado de YOLO26 segmentación

YOLO26-seg reporta hasta +3.7 mask AP sobre YOLO11 en COCO, con un módulo proto
mejorado que usa información multiescala y pérdida de segmentación semántica
auxiliar. Es una elección razonable. Los puntos de atención documentados por la
comunidad son los de H-06 (duplicados en modo end2end) y los problemas históricos
de `retina_masks` con padding no divisible por 2
([ultralytics#6110](https://github.com/ultralytics/ultralytics/issues/6110)) —
este último ya está mitigado aquí porque el wrapper valida `orig_shape` contra el
tamaño de la imagen antes de rasterizar.

### TTA

`model.predict(augment=True)` mejora recall en varios contextos, pero hay
evidencia de degradación cuando el objeto es pequeño respecto a la imagen, porque
las escalas descendentes de TTA empeoran objetos ya reducidos. En este dominio la
hoja objetivo suele ocupar >25 % del cuadro, así que el riesgo es bajo y el
beneficio plausible sobre el subgrupo de hoja pequeña. Vale como ablación de
inferencia (sin reentrenar), medida sobre `val` y sobre el subgrupo `small_leaf`
de la auditoría. Coste: ~3× tiempo de inferencia, relevante si el destino es
móvil.

---

## Plan de acción priorizado

| # | Acción | Hallazgos | Esfuerzo | Bloquea |
|---|---|---|---|---|
| 1 | Resolver el contrato 183 vs 182 instancias en `test` | H-17 | Bajo | Evaluación final |
| 2 | Despinnear el checkpoint de `evaluate_mode` vía manifiesto de promoción | H-02 | Bajo | Evaluación final |
| 3 | Alinear defaults del gate con la calibración y centralizar el loader de config | H-03 | Bajo | Consumo externo correcto |
| 4 | Fijar `checkpoint_sha256` en la ruta de inferencia | H-12 | Bajo | — |
| 5 | Separar `val_metrics` de `pipeline_metrics`; extraer evaluador de pipeline reutilizable | H-01 | Medio | Comparabilidad de métricas |
| 6 | Dedupe de instancias por mask-IoU + registrar `end2end` real | H-06 | Bajo | Cobertura del gate |
| 7 | Normalizar `normalized_perimeter` por canvas y añadir señales ortogonales al gate | H-04, H-05 | Medio | Robustez del gate |
| 8 | Corregir `crop_mask_letterbox` (NEAREST + remask) | H-07 | Bajo | Calidad de salida |
| 9 | Ablaciones `D-07` (`mask_ratio=1`) y `D-08` (rotación/flip vertical) | H-10 | Medio | Calidad de contorno |
| 10 | Repetir D-01 con semillas 7 y 1337, reportar media ± σ | — | Medio | Promoción del modelo |
| 11 | Ronda de 300 imágenes difíciles, usada además como validación del gate | H-11, brecha de dominio | Alto | Todo lo demás |
| 12 | Higiene: importación diferida de torch, `make check` con lint+format, tests herméticos | H-14, H-15, H-16 | Bajo | — |

Las acciones 1 a 4 son de bajo esfuerzo y desbloquean el resto; conviene hacerlas
en un solo bloque antes de gastar la ejecución única de `test`.

## Estado de verificación al momento de la revisión

```
python -m pytest        297 passed, 2 failed  (fallos por entorno: ver H-16)
                        3 módulos no colectables sin torch
python -m ruff check .  All checks passed
python -m ruff format --check .  47 archivos se reformatearían
```

## Fuentes

- [Boundary IoU: Improving Object-Centric Image Segmentation Evaluation (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf)
- [Masking Strategies for Background Bias Removal in Computer Vision Models](https://arxiv.org/abs/2308.12127)
- [The Impact of Background Removal on Performance of Neural Networks](https://arxiv.org/html/2308.09764v2)
- [Uncovering bias in the PlantVillage dataset](https://arxiv.org/abs/2206.04374)
- [Quantifying the reliability gap in cross-domain plant disease classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC13236948/)
- [Deep Learning Diagnostics of Gray Leaf Spot in Maize under Mixed Disease Field Conditions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330607/)
- [Deep learning-based segmentation and classification of leaf images for tomato plant disease](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1031748/full)
- [Semantic segmentation for plant leaf disease classification and damage detection](https://www.sciencedirect.com/science/article/pii/S277237552400131X)
- [Orchestrating segment anything models to accelerate segmentation annotation on agricultural image datasets](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1748468/full)
- [An Automated Image Segmentation, Annotation, and Training Framework of Plant Leaves by Joining SAM and YOLOv8](https://doi.org/10.3390/agronomy15051081)
- [Towards precision agriculture: A dataset for early detection of corn leaf pests](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11905841/)
- [Mobile-assisted deep learning framework for identification of insect pests and diseases of maize from field images](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1803005/full)
- [Ultralytics YOLO26: por qué se elimina NMS](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment)
- [ultralytics#23685 — yolo26*-seg end2end duplicate detections](https://github.com/ultralytics/ultralytics/issues/23685)
- [ultralytics#6110 — retina_masks y padding no divisible por 2](https://github.com/ultralytics/ultralytics/issues/6110)
- [Ultralytics configuration reference (defaults de `conf` por modo)](https://docs.ultralytics.com/usage/cfg/)
- [Test-Time Augmentation (Ultralytics)](https://docs.ultralytics.com/yolov5/tutorials/test-time-augmentation)
