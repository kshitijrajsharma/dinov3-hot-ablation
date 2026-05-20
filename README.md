# dinov3-hot

Binary building footprint segmentation from VHR RGB aerial imagery, built on a frozen DINOv3-ViT-L/16 (LVD1689M) encoder with a UperNet decoder, multi-task auxiliary heads, and watershed instance separation. Trained on the global [hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation) dataset; ships to [hotosm/fAIr-models](https://github.com/hotosm/fAIr-models) as a portable ONNX model.

## Approach

The encoder stays frozen. We learn a UperNet decoder on top of it, plus three small task-specific heads that share supervision signal:

| Head | Output | Why it exists |
| --- | --- | --- |
| Mask | `sigmoid` of channel 0 | The thing we ship: is this pixel a building? |
| Boundary | `sigmoid` of channel 1 | A 2-pixel ring around each building polygon; sharpens edges where buildings touch. |
| Distance | `tanh` of channel 2 | Signed distance to the nearest boundary, clipped to +/-15 px and normalized to [-1, 1]. Provides watershed seeds at inference. |

An auxiliary FCN head taps the third encoder layer's features and predicts the same three channels with loss weight 0.4. It is supervision-only and is dropped at inference time.

```mermaid
flowchart TD
    A[RGB tile<br/>3 x 256 x 256] --> B[DINOv3-L encoder, frozen<br/>get_intermediate_layers norm=True]
    B --> C{taps at layers<br/>5, 11, 17, 23}
    C --> D[LearnedInterpolateToPyramidal<br/>scales 4x, 2x, 1x, 0.5x]
    D --> E[UperNetDecoder<br/>PPM + FPN, 512 ch]
    E --> F[1x1 Conv head<br/>3 channels]
    C -. tap-2, training only .-> G[Aux FCN head<br/>3 channels, loss x 0.4]
    F --> H[mask logit, boundary logit, signed distance]
    H --> I[sigmoid mask > 0.5<br/>+ watershed seeded by peak_local_max on distance<br/>+ vectorize + Douglas-Peucker 0.5 m]
    I --> J[GeoJSON polygons]
```

At inference time, the predicted mask + distance map go through a watershed pass: local maxima of the distance map become per-building seeds, watershed assigns each pixel to its nearest seed, and the result is vectorized with Douglas-Peucker simplification at 0.5 m. This produces clean, instance-separated polygons in dense urban scenes where naive connected components would merge neighboring buildings.

The recipe follows Meta's reference UperNet recipe for DINOv3 ([github issue #54](https://github.com/facebookresearch/dinov3/issues/54)) with two project-specific additions: the boundary + distance heads, and the watershed post-process.

## Loss

The training objective is a weighted sum of four terms applied to the main head's three output channels, plus the same combination applied to the auxiliary head's outputs at weight 0.4:

```
loss = BCE(mask)
     + Dice(mask)
     + alpha * BCE(boundary)
     + beta  * Huber(distance, delta=1.0)
     + gamma * TV(sigmoid(mask))
     + 0.4 * <same combination on aux head>
```

The weights `alpha`, `beta`, `gamma` are found by Optuna TPE hyperparameter search. The TV term (`kornia.losses.total_variation`) penalizes pixel-to-pixel jaggedness on the mask probability; HPO can pick zero if it doesn't help.

The shipped checkpoint uses the params in [`conf/experiments/v5_hpo_best.yaml`](conf/experiments/v5_hpo_best.yaml): `alpha=0.27`, `beta=0.47`, `gamma=0.049`, `aux_loss_weight=0.22`.

## Metrics

Two distinct metrics, both reported separately:

### Pixel IoU (binary Jaccard)

`torchmetrics.classification.BinaryJaccardIndex` aggregated across all evaluation pixels. Standard intersection-over-union on the binary mask. Tells us "what fraction of foreground pixels did we correctly predict?". This is what we monitor during training (`val/iou`) and report on the HF test split (`test/iou`).

Pixel IoU is the right metric when "is this pixel a building" is the question. It does not say anything about whether neighboring buildings got separated or merged.

### Instance F1 @ IoU > 0.5

The harmonic mean of instance precision and recall, where each predicted polygon is matched to at most one ground-truth polygon by Hungarian matching with pixel-IoU > 0.5 as the assignment criterion. Implemented in [`src/dinov3_hot/metrics.py`](src/dinov3_hot/metrics.py) on top of `torchmetrics.detection.PanopticQuality` (which already does this matching) with a thin wrapper that exposes precision and recall separately.

This is **not** mAP. mAP integrates precision-recall across confidence thresholds and is used for object detection benchmarks like COCO. Instance F1 at a fixed IoU threshold is the standard metric in the cell-segmentation and building-footprint literature (stardist, Cellpose, SpaceNet) because the question is "did we recover each building as its own polygon?", not "rank these polygons by confidence".

For dense urban building segmentation, **instance F1 is the metric that matches the user's intent**. Pixel IoU can stay high while instance F1 collapses if the model merges touching buildings into single blobs.

We report both:

- **Pixel IoU**: how much of the building area was found.
- **Instance F1**: how often individual buildings were recovered as distinct polygons.

### What about mAP?

mAP would need (a) a confidence score per predicted polygon, and (b) a way to vary the score threshold and trace a precision-recall curve. Our pipeline outputs binary masks then vectorizes; there's no natural per-polygon score to threshold on, so mAP doesn't apply cleanly. If we needed score-ranked output we'd compute it from average pixel-prob within each polygon, but for fAIr's end use case (mapper-grade polygons) the user wants a single threshold, so instance F1 is the right summary.

## Results

Banepa OAM (Nepali dense urban, 1536x1536 raster, 2720 OSM ground-truth polygons):

| Variant | Polygons | Pixel IoU | Precision | Recall | Instance F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1+FT (n-1 release) | 3523 | 0.680 | 0.333 | 0.432 | 0.376 |
| **v5 (this release, no per-area FT)** | **1984** | **0.667** | **0.493** | **0.360** | **0.416** |

+4 pp instance F1 over the previous shipped model, with no per-area finetune required. Precision jumps by 16 pp (0.333 -> 0.493): v5's polygons are dramatically more likely to be real buildings.

On the HF test split (global, heterogeneous), v5 reports pixel IoU 0.441.

## Layout

```
src/dinov3_hot/        # Python package: model, data, train, infer, finetune, hpo, export, eval, metrics
conf/                  # YAML configs
conf/experiments/      # tracked snapshots of HPO-found params per release
scripts/               # one-off analysis scripts (Banepa eval, geometry viz)
tests/                 # pytest suite
outputs/               # local run artifacts (gitignored)
pr_pack/               # fAIr-models drop (gitignored; staged locally, push to fAIr-models repo)
```

## Usage

```bash
# install
just setup

# train at 100% data with HPO
uv run dinov3-hot train --config conf/train.yaml

# train with HPO disabled (uses fixed cfg values)
uv run dinov3-hot train --config conf/train.yaml hpo.enabled=false

# sliding-window inference on a GeoTIFF
uv run dinov3-hot predict --ckpt outputs/dinov3l_v5/ckpts/best-05-0.5809.ckpt \
  --raster path/to/raster.tif --out path/to/predictions.geojson

# per-area decoder finetune on a small chip set
uv run dinov3-hot finetune --ckpt <path> --chips-dir <path> \
  --labels-geojson <path> --out-dir <path>

# export to ONNX for fAIr-models deployment
uv run dinov3-hot export --ckpt <path> --out pr_pack/dinov3_buildings/artifacts/dinov3_buildings.onnx
```

## Stack

- Python 3.13, `uv` package manager
- PyTorch 2.7, PyTorch Lightning 2.6
- `terratorch` for the UperNet decoder and `LearnedInterpolateToPyramidal` neck
- `torchmetrics` for IoU and PanopticQuality-based instance matching
- `optuna` for HPO
- `rasterio`, `geopandas`, `shapely`, `skimage`, `scipy` for geospatial I/O and post-processing
- `kornia` for total-variation loss
- `segmentation_models_pytorch` for Dice loss
- `huggingface_hub` and `datasets` for backbone weights and training data

## License

Apache-2.0. DINOv3-L encoder weights from Facebook Research, Apache-2.0.
