# Entrenamiento cloud del segmentador

El entrenamiento de `yolo26n-seg.pt` queda reservado a una máquina CUDA
remota. El equipo local sólo construye y valida un paquete reproducible; no
instala Ultralytics, no descarga pesos y no ejecuta forward ni épocas.

## Contenido

`cloud_training/` contiene:

- bootstrap con resolución `pip --dry-run`, constraints dinámicos y bloqueo si
  se intenta sustituir torch o torchvision;
- preflight GPU/modelo que vuelve a verificar locks, fingerprints canónicos y
  dataset, y ejecuta un forward sintético del head de segmentación sin depender
  de que una imagen produzca detecciones;
- smoke de una época protegido por
  `CONFIRM_SEGMENTATION_SMOKE_TRAINING=1`;
- entrenamiento y reanudación protegidos por
  `CONFIRM_SEGMENTATION_TRAINING=1`;
- evaluación final de un solo uso sobre el test retenido, con comprobación del
  split efectivo reportado por Ultralytics;
- configs YAML, manifiesto del paquete y checksums.

El payload incluye únicamente `images/{train,val,test}`,
`labels/{train,val,test}`, `dataset.yaml`, los cinco manifiestos necesarios,
código, scripts y documentación mínima. Excluye `all/`, fuentes externas,
piloto, ZIP, outputs históricos, checkpoints, notebooks, entornos y cachés.

El piloto tiene un manifiesto de transporte separado y no forma parte del
archivo de entrenamiento.

## Transporte hacia Modal

El dataset congelado se publica una sola vez en Hugging Face con
`leaf-segmentation-hf-publish` y `modal_training.py::seed_dataset` lo materializa
dentro del Volume `doctor-maiz-leaf-segmentation-data`. El código viaja en la
imagen de Modal, de modo que iterarlo no obliga a resubir el dataset.

La identidad del dataset no la da el transporte: antes de cada operación remota,
`verify_cloud_training_payload` recalcula el fingerprint padre, los tres de split
y los conteos sobre el árbol montado, y deja constancia en
`outputs/leaf_detection/modal_runtime/dataset_verification.json`.

Los artefactos viven en el Volume `doctor-maiz-leaf-segmentation-outputs` y se
bajan completos con `leaf-segmentation-modal-download`.

## Flujo remoto

```bash
make leaf-segmentation-cloud-bootstrap
make leaf-segmentation-cloud-preflight

CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 \
  make leaf-segmentation-cloud-smoke

# Revise el batch medido y congele cualquier ajuste de workers/cache en este YAML.
CONFIRM_SEGMENTATION_TRAINING=1 \
  make leaf-segmentation-cloud-train \
  CONFIG=outputs/leaf_detection/segmenter/configs/train_yolo26n_seg.final.yaml

make leaf-segmentation-cloud-validate
```

## Flujo con Modal

El baseline `yolo26n_seg_baseline` ya está completo y se conserva como control.
La release `v7-segmentation-improvements-7a4a5c08-seed42` prepara experimentos
separados, de modo que cada corrida tiene su propio directorio, resumen y
manifiesto y no puede sobrescribir el baseline.

La primera corrida de esa release, `d01_mosaic0_seed42`, terminó correctamente
el 2026-08-17. Usó A10, batch 26 y parada temprana después de 145 épocas; la
mejor fue la 115. El checkpoint local y remoto comparten SHA-256
`a2bf4f201ca4f5e32c349cdc66d7ac39a6b012a330b182149401e533b2ecb8ab`.
Los resultados completos están en
[Resultados de D-01](segmentation-d01-results.md).

```bash
# Operaciones sin épocas de entrenamiento.
make leaf-segmentation-modal-seed
make leaf-segmentation-modal-verify-dataset
make leaf-segmentation-modal-preflight MODAL_SEGMENTATION_GPU=A10

# Primera ablación recomendada: desactivar mosaic.
CONFIRM_SEGMENTATION_TRAINING=1 \
  make leaf-segmentation-modal-experiment \
  MODAL_SEGMENTATION_EXPERIMENT=d01_mosaic0_seed42 \
  MODAL_SEGMENTATION_GPU=A10

make leaf-segmentation-modal-results
make leaf-segmentation-modal-checksums
make leaf-segmentation-modal-download
```

Los artefactos canónicos descargados quedan bajo
`outputs/leaf_detection/segmenter/d01_mosaic0_seed42/` y su resumen bajo
`outputs/leaf_detection/segmenter/experiment_summaries/`. Las evaluaciones
locales oficiales deben usar un nombre propio bajo `segmenter/evaluations/` y
registrar el hash del checkpoint y del manifiesto evaluado.

Los perfiles permitidos directamente son `d01_mosaic0_seed42`,
`d02_imgsz512_seed42`, `d03_source_balanced_seed42`, `d05_scratch_seed42`,
`d06_copy_paste_seed42`, `e01_baseline_seed7` y
`e01_baseline_seed1337`. Los perfiles `d02b_imgsz768_seed42` y
`d04_yolo26s_seed42` están bloqueados hasta ejecutar un smoke específico; el
segundo también necesita pesos `yolo26s-seg.pt` registrados y verificados.

Si una ablación queda en estado `running` o `failed` y conserva `last.pt`, se
reanuda explícitamente con:

```bash
CONFIRM_SEGMENTATION_TRAINING=1 \
  make leaf-segmentation-modal-experiment-resume \
  MODAL_SEGMENTATION_EXPERIMENT=d01_mosaic0_seed42 \
  MODAL_SEGMENTATION_GPU=A10
```

La evaluación final tiene un gate de un solo uso: `validate.sh` aborta si
`test_summary.json` ya existe. Envía `split=test` explícitamente y el runner
bloquea si el validador de Ultralytics reporta cualquier otro split. Antes de
evaluar también exige 173 imágenes, 183 instancias, el fingerprint de test
congelado y el SHA-256 exacto de `best.pt`.

La evaluación está temporalmente bloqueada: Ultralytics 8.4.104 conserva 182
instancias efectivas de las 183 anotaciones canónicas porque dos polígonos
distintos de `cldc_ec40ec2d7da5243e.txt` comparten bbox. No se publicó un
resumen aprobado y se eliminaron todas las carpetas de evaluación `val/test`
generadas durante el diagnóstico. No se debe reintentar hasta resolver ese
contrato sin modificar el fingerprint test.

Cada reanudación escribe su propio `resume_manifest_<timestamp>.json` además de
la copia con nombre estable, de modo que una corrida interrumpida varias veces
conserva el historial completo.

El preflight puede resolver los pesos en la nube, registra su ruta, tamaño,
SHA-256, fuente y versión exacta de Ultralytics, verifica el task y el head de
segmentación con un tensor CUDA sintético y se detiene antes del entrenamiento.
El smoke usa `batch=-1` para medir AutoBatch y sólo declara éxito si el batch
efectivo es un entero positivo, las pérdidas/métricas son finitas, la GPU fue
usada y `last.pt` existe con hash. La configuración final se crea una vez bajo
`outputs/leaf_detection/segmenter/configs/`; nunca se sobrescribe.

Antes de iniciar las 150 épocas, el runner crea
`active_run_manifest.json` con la identidad, configuración, fingerprints y
rutas esperadas del run. Así una interrupción se reanuda desde el `last.pt`
exacto y nunca desde un directorio incrementado implícitamente.

La reanudación nunca es automática:

```bash
CONFIRM_SEGMENTATION_TRAINING=1 \
  make leaf-segmentation-cloud-resume
```

## Construcción local permitida

```bash
make leaf-segmentation-cloud-package
```

El empaquetador usa lista blanca, orden estable, UID/GID cero, permisos
normalizados y mtime cero. Genera el `.tar.gz`, su `.sha256`, extrae en un
temporal, valida cada checksum y confirma que ninguna ruta excluida aparezca.

No se modifica ningún proyecto consumidor del segmentador. No se usa el
piloto hasta una fase externa posterior.
