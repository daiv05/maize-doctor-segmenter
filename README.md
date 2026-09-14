# DoctorMaiz Leaf Segmentation

Proyecto independiente para preparar, entrenar, auditar y ejecutar un segmentador de hojas de maíz basado en YOLO instance segmentation. El repositorio no contiene el clasificador de enfermedades: ese modelo y sus pipelines viven en [`maize-doctor-classifier`](https://github.com/daiv05/maize-doctor-classifier), que consume este segmentador para derivar una máscara foliar.

## Alcance

- consolidación de fuentes YOLO/COCO y revisión humana;
- normalización JPEG y trazabilidad por SHA-256;
- splits agrupados y reproducibles, sin fugas contra el piloto retenido;
- preflight, empaquetado cloud, entrenamiento, reanudación y evaluación YOLO;
- inferencia del segmentador, selección determinista de la hoja y salidas de máscara;
- métricas IoU/Dice/recall, calibración y auditoría del quality gate.

Los datos materializados viven en `data/leaf_detection/`; los checkpoints, paquetes, predicciones y reportes viven en `outputs/leaf_detection/` y no se versionan.

## Estado

| | |
|---|---|
| Dataset congelado | 1 155 imágenes y 1 224 máscaras; lock `7a4a5c08` |
| Splits agrupados | 809 / 173 / 173 sobre 1 035 grupos |
| Piloto retenido | 100 imágenes, cero fugas verificadas, nunca entra en train/val/test |
| Checkpoint promovido | corrida `yolo26n_seg_baseline`, registrada en `outputs/leaf_detection/models/promoted_checkpoint.json` |
| Candidato D-01 (`mosaic=0.0`) | Mask mAP50-95 `0.94404` en `val`, mejor época 115 |
| Validación sobre hojas enfermas | 150/150 reliable, IoU `0.98122` |
| Evaluación final sobre `test` | bloqueada; ver [auditoría](docs/es/leaf-detection/segmentation-pipeline-deep-review.md) |

D-01 es el candidato actual, no el promovido: mejora el Mask mAP50-95 del baseline en 0.59 puntos porcentuales, pero esa comparación usa `val` y todavía no demuestra estabilidad entre semillas. Los detalles reproducibles, con hashes de paquete, configuración y pesos, están en [resultados de D-01](docs/es/leaf-detection/segmentation-d01-results.md).

## Inicio rápido

```bash
git clone https://github.com/abner-rivas/maize-doctor-segmenter-leaf
cd maize-doctor-segmenter-leaf
python -m venv .venv
source .venv/bin/activate
cp .env.example .env
make install
make leaf-segmentation-status
make leaf-segmentation-verify-locks
make leaf-segmentation-verify-splits
```

El entrenamiento nunca se inicia implícitamente. Los targets de smoke, train y resume exigen las confirmaciones literales mostradas por `make help`.

Variables de entorno y requisitos en [LOCAL.md](LOCAL.md).

## Perfiles de salida

`src/preprocessing/segmented_leaf_processor.py` decide qué se le entrega al clasificador una vez seleccionada la hoja. El perfil se fija en `config/segmentation.yaml` bajo `output_profile`:

| Perfil | Salida |
|---|---|
| `mask_black` | Imagen completa con todo lo que no es hoja en negro |
| `bbox_crop` | Recorte al bounding box de la hoja, sin enmascarar |
| `crop_mask_black` | Recorte al bounding box con el fondo enmascarado en negro |
| `crop_mask_letterbox` | Igual que el anterior, ajustado a `target_size` con letterbox |
| `square_crop` | Recorte cuadrado centrado en la hoja, conservando el fondo natural |

`fallback` decide qué ocurre cuando el quality gate rechaza la máscara: `original` devuelve la imagen sin tocar, `reject` la descarta.

## Estructura

```text
cloud_training/   runner y scripts de entrenamiento YOLO en CUDA
config/           configuración de inferencia y quality gate del segmentador
data/             dataset segmentado, piloto, locks y manifiestos (ignorado salvo README)
docs/es/          decisiones, auditorías y flujo técnico de segmentación
scripts/checks/   validadores de ROI y manifiestos
scripts/dataset/  consolidación, revisión, finalización y splits
scripts/experiments/ calibración del quality gate, del umbral de selección y auditoría
scripts/package/  paquete cloud determinista y verificadores
scripts/pipeline/ preflight, evaluación, piloto y promoción de checkpoint
src/data/         lógica reproducible del dataset de segmentación
src/evaluation/   métricas downstream, calibración y reliability audit
src/preprocessing/geometría, máscara, ROI y letterbox
src/segmentation/ adaptador YOLO y quality gate
src/training/     preflight, experimentos y promoción de checkpoints
tests/            pruebas exclusivas del flujo de segmentación
```

## Comandos principales

```bash
make help
make leaf-segmentation-preflight
make leaf-segmentation-downstream-metrics PREDICTIONS=<directorio>
make leaf-segmentation-reliability-audit
make leaf-segmentation-calibrate-quality-gate
make leaf-segmentation-calibrate-selection
make leaf-segmentation-pilot-evaluate
make leaf-segmentation-promote-checkpoint

CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 make leaf-segmentation-cloud-smoke
CONFIRM_SEGMENTATION_TRAINING=1 make leaf-segmentation-cloud-train
```

El dataset congelado se publica y se recupera desde Hugging Face:

```bash
make leaf-segmentation-hf-publish HF_SEGMENTATION_STAGE_DIR=<directorio con ~2.4 GB>
make leaf-segmentation-hf-download
```

El entrenamiento remoto siembra ese dataset en un Volume de Modal una sola vez:

```bash
make leaf-segmentation-modal-seed
make leaf-segmentation-modal-verify-dataset
make leaf-segmentation-modal-preflight
CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 make leaf-segmentation-modal-smoke
CONFIRM_SEGMENTATION_TRAINING=1 make leaf-segmentation-modal-train
CONFIRM_SEGMENTATION_TRAINING=1 make leaf-segmentation-modal-experiment MODAL_SEGMENTATION_EXPERIMENT=<perfil>
make leaf-segmentation-modal-validate
make leaf-segmentation-modal-promote
make leaf-segmentation-modal-download
```

## Documentación

- [Flujo actual](docs/es/leaf-detection/segmentation-current-flow.md) — estado por etapa, verificado contra locks y artefactos.
- [Revisión profunda del pipeline](docs/es/leaf-detection/segmentation-pipeline-deep-review.md) — auditoría técnica y hallazgos abiertos.
- [Resultados de D-01](docs/es/leaf-detection/segmentation-d01-results.md) — la ablación candidata y su prueba sobre hojas enfermas.
- [Protocolo de mejora](docs/es/leaf-detection/segmentation-improvement-protocol.md) — cómo se eligen y validan los experimentos.
- [Decisiones de arquitectura](docs/es/decisions/adr-leaf-instance-segmentation-strategy.md) — ADRs del proyecto.

## Licencia

El código de este repositorio se distribuye bajo la licencia MIT. Ver [LICENSE](LICENSE).

El extra opcional `segmentation` instala `ultralytics==8.4.104`, publicada bajo **AGPL-3.0**. MIT cubre el código propio, no las dependencias: distribuir un trabajo combinado con Ultralytics, o exponerlo como servicio en red, activa las obligaciones de la AGPL. Para un uso que no pueda asumirlas hace falta la licencia comercial de Ultralytics o sustituir esa dependencia. Los datasets externos consolidados conservan además sus propias licencias de origen.

Proyecto académico desarrollado en la Universidad de El Salvador.
