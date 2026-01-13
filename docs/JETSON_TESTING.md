# Jetson testing (container + inference FPS)

Obiettivo: eseguire la pipeline (detector + tracker) su una macchina NVIDIA Jetson in container (basato su un’immagine fornita) e misurare throughput/latency per stimare la fattibilità su hardware “edge”.

## 1) Prerequisiti sulla Jetson
- JetPack installato e GPU funzionante
- Docker + NVIDIA Container Runtime (`nvidia-container-toolkit`)
- Dataset montato localmente (nuScenes export MOT front) e pesi disponibili

> Nota: in molti setup Jetson, `torch/torchvision` sono già inclusi nell’immagine base. In caso contrario, vanno installati con le wheel compatibili con JetPack.

## 2) Costruire l’immagine (estendendo l’immagine fornita)
Da root del workspace (cartella che contiene `tesi_project_amarino/`):

```bash
docker build -f tesi_project_amarino/jetson/Dockerfile \
  --build-arg BASE_IMAGE=<IMMAGINE_FORNITA_DAL_PROF> \
  -t track-bench:jetson .
```

## 3) Avvio container con mount di dataset e pesi
Esempio (adatta i path):

```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  -v /path/to/nuscenes_mot_front:/workspace/tesi_project_amarino/data/nuscenes_mot_front \
  -v /path/to/weights:/workspace/tesi_project_amarino/weights \
  track-bench:jetson
```

## 4) Benchmark FPS/latency
`track.py` supporta la modalità benchmark (timing per frame e report JSON).

Esempio TrackSSM (config finale, senza salvare i risultati di tracking per evitare overhead I/O):

```bash
cd /workspace/tesi_project_amarino
python3 track.py \
  --tracker trackssm \
  --data data/nuscenes_mot_front/val \
  --output results/JETSON_TRACKSSM \
  --detector-weights yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth \
  --conf-thresh 0.5 --nms-thresh 0.6 \
  --match-thresh 0.8 --track-thresh 0.6 --max-age 30 \
  --benchmark \
  --benchmark-warmup 5 \
  --benchmark-max-frames 0 \
  --benchmark-sync-cuda \
  --no-save-results
```

Esempio BoT-SORT:

```bash
python3 track.py \
  --tracker botsort \
  --data data/nuscenes_mot_front/val \
  --output results/JETSON_BOTSORT \
  --detector-weights yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth \
  --conf-thresh 0.5 --nms-thresh 0.6 \
  --match-thresh 0.8 --track-thresh 0.6 --max-age 30 \
  --benchmark \
  --benchmark-warmup 5 \
  --benchmark-max-frames 0 \
  --benchmark-sync-cuda \
  --no-save-results
```

Output:
- `results/.../benchmark.json` contiene:
  - stats globali (FPS medio, median/p95 latenza)
  - breakdown per scena
  - breakdown per fase (`read_ms`, `detect_ms`, `track_ms`, `total_ms`)

## 5) Note per un benchmark “pulito”
- Usare `--benchmark-sync-cuda` per tempi realistici su GPU
- Disabilitare `--save-videos` e (in generale) scritture non necessarie
- Se il dataset è su storage lento, il tempo di I/O può dominare: in quel caso valuta anche `--benchmark-include-io false` (se vuoi misurare solo compute)
