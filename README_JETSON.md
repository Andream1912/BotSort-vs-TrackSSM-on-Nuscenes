# Jetson: setup + benchmark (TrackSSM / BoT-SORT)

Questa guida serve per replicare i test su **NVIDIA Jetson** dopo aver fatto il `git clone` del repository e aver copiato **(1)** YOLOX fine-tuned e **(2)** TrackSSM fine-tuned nel progetto.

> Nota licenze/dimensione: il dataset nuScenes **non va pushato su GitHub** (licenza + dimensioni). Va scaricato dal sito ufficiale o trasferito via Drive.

## 0) Prerequisiti

- JetPack installato (CUDA funzionante)
- `git`, `python3`, `pip`
- PyTorch + CUDA compatibile JetPack (consigliato: wheel NVIDIA ufficiale)

Verifica GPU:

```bash
python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY
```

## 1) Clone del repo

```bash
git clone git@github.com:Andream1912/BotSort-vs-TrackSSM-on-Nuscenes.git
cd BotSort-vs-TrackSSM-on-Nuscenes
```

## 2) Dipendenze Python

Dentro `tesi_project_amarino/`:

```bash
cd tesi_project_amarino
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

Se qualche package “pesante” fallisce in build su Jetson (es. `mamba_ssm`), la soluzione corretta dipende dalla tua versione JetPack/PyTorch: usa wheel/prebuild compatibili o compila dal sorgente.

## 3) Dipendenze esterne (YOLOX)

Il progetto si aspetta YOLOX in `external/YOLOX`.

```bash
mkdir -p external
git clone https://github.com/Megvii-BaseDetection/YOLOX external/YOLOX
```

## 3.1) Dipendenze esterne (BoT-SORT)

Il progetto si aspetta BoT-SORT in `external/BoT-SORT`.

```bash
mkdir -p external
git clone https://github.com/NirAharon/BoT-SORT external/BoT-SORT
```

## 4) Copia dei pesi (da Drive)

Copia i due checkpoint nei path attesi (o aggiorna i path nei comandi).

### YOLOX fine-tuned
Atteso qui:

- `tesi_project_amarino/yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth`

Esempio:

```bash
mkdir -p yolox_finetuning/yolox_l_nuscenes_stable
cp /path/drive/epoch_30.pth yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth
```

### TrackSSM fine-tuned (Phase2)
Atteso qui:

- `tesi_project_amarino/weights/trackssm/phase2/phase2_full_best.pth`

Esempio:

```bash
mkdir -p weights/trackssm/phase2
cp /path/drive/phase2_full_best.pth weights/trackssm/phase2/phase2_full_best.pth
```

## 5) Dataset (nuScenes MOT front)

Struttura attesa (MOT-style):

- `tesi_project_amarino/data/nuscenes_mot_front/val/scene-XXXX_CAM_FRONT/img1/*.jpg`
- `tesi_project_amarino/data/nuscenes_mot_front/val/scene-XXXX_CAM_FRONT/gt/gt.txt`

Se non lo hai localmente su Jetson: sì, mettilo su Drive (o trasferiscilo in altro modo) e copialo/montalo sotto `data/nuscenes_mot_front/`.

## 6) Smoke test (una scena)

```bash
python3 track.py \
  --tracker trackssm \
  --data data/nuscenes_mot_front/val \
  --output results/JETSON_SMOKE \
  --device cuda \
  --detector-weights yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth \
  --conf-thresh 0.5 --nms-thresh 0.6 \
  --no-reid \
  --cmc-method none \
  --trackssm-batch \
  --scenes scene-0520_CAM_FRONT \
  --benchmark --benchmark-warmup 5 --benchmark-sync-cuda \
  --no-save-results
```

Output atteso:
- `results/JETSON_SMOKE/benchmark.json`

## 7) Config “fast” vs “best metrics”

- **FAST (più FPS)**: `--cmc-method none --trackssm-batch`
- **BEST METRICS**: `--cmc-method sparseOptFlow --trackssm-batch`

Il batching (`--trackssm-batch`) è la principale ottimizzazione implementata: riduce drasticamente la latenza del tracker evitando forward separati per ogni track.

## 8) Note utili

- Per timing realistici su GPU: usa `--benchmark-sync-cuda`.
- Per misurare solo compute: lascia `--benchmark-include-io` disattivo (default).
- Evita I/O: `--no-save-results` e niente `--save-videos`.

(Guida container opzionale: vedi `docs/JETSON_TESTING.md`.)
