# Jetson: setup + benchmark (TrackSSM / BoT-SORT)

Questa guida serve per replicare i test su **NVIDIA Jetson** dopo aver fatto il `git clone` del repository e aver copiato **(1)** YOLOX fine-tuned e **(2)** TrackSSM fine-tuned nel progetto.

> Nota licenze/dimensione: il dataset nuScenes **non va pushato su GitHub** (licenza + dimensioni). Va scaricato dal sito ufficiale o trasferito via Drive.

## Scelta rapida: nativo o Docker?

Hai due strade. Se non hai confidenza con Docker, ti consiglio **setup nativo** (più semplice da debuggare su Jetson).

- **A) Setup nativo (consigliato)**: installi dipendenze sul sistema/venv e lanci `track.py` direttamente.
- **B) Docker (opzionale)**: ambiente isolato; utile se vuoi replicabilità e non “sporcare” il sistema.

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

Verifica versione JetPack (ti serve se userai un’immagine Docker base o wheel PyTorch giusta):

```bash
head -n 1 /etc/nv_tegra_release || true
```

---

## A) Setup nativo (consigliato)

### A1) Aggiorna pacchetti di sistema

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg python3-venv python3-dev
```

### A2) Clone del repo

```bash
git clone git@github.com:Andream1912/BotSort-vs-TrackSSM-on-Nuscenes.git
cd BotSort-vs-TrackSSM-on-Nuscenes/tesi_project_amarino
```

### A3) Crea un environment (venv)

Sì: su Jetson è consigliato usare un environment per evitare conflitti.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

### A4) PyTorch su Jetson (passo fondamentale)

Il modo “giusto” dipende dal JetPack. Prima controlla se `torch` è già installato e vede CUDA:

```bash
python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY
```

Se `cuda available` è `False` o `torch` manca, installa una wheel PyTorch **compatibile con JetPack** (tipicamente NVIDIA). Questa parte è la più delicata: segui la documentazione NVIDIA per la tua release.

### A5) Installa dipendenze Python (Jetson-friendly)

Su Jetson evitiamo di reinstallare `torch/torchvision/opencv-python` via pip (di solito sono già forniti dal sistema/base wheel).

```bash
pip3 install -r requirements_jetson.txt
```

### A6) Dipendenze esterne (YOLOX + BoT-SORT)

Il progetto si aspetta:
- YOLOX in `external/YOLOX`
- BoT-SORT in `external/BoT-SORT`

```bash
mkdir -p external
git clone https://github.com/Megvii-BaseDetection/YOLOX external/YOLOX
git clone https://github.com/NirAharon/BoT-SORT external/BoT-SORT
```

### A7) Copia dei pesi (da Drive/PC)

Questi file **non** sono nel repo: vanno copiati manualmente.

YOLOX fine-tuned (atteso):
- `yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth`

TrackSSM Phase2 fine-tuned (atteso):
- `weights/trackssm/phase2/phase2_full_best.pth`

Esempio:

```bash
mkdir -p yolox_finetuning/yolox_l_nuscenes_stable
mkdir -p weights/trackssm/phase2

cp /path/drive/epoch_30.pth yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth
cp /path/drive/phase2_full_best.pth weights/trackssm/phase2/phase2_full_best.pth
```

Suggerimento pratico (da PC → Jetson):

```bash
# sul PC
scp epoch_30.pth user@JETSON_IP:/path/BotSort-vs-TrackSSM-on-Nuscenes/tesi_project_amarino/yolox_finetuning/yolox_l_nuscenes_stable/
scp phase2_full_best.pth user@JETSON_IP:/path/BotSort-vs-TrackSSM-on-Nuscenes/tesi_project_amarino/weights/trackssm/phase2/
```

### A8) Dataset (nuScenes MOT front)

Struttura attesa (MOT-style):

- `data/nuscenes_mot_front/val/scene-XXXX_CAM_FRONT/img1/*.jpg`
- `data/nuscenes_mot_front/val/scene-XXXX_CAM_FRONT/gt/gt.txt`

### A9) Smoke test (una scena)

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
  --scenes scene-0014_CAM_FRONT \
  --benchmark --benchmark-warmup 5 --benchmark-sync-cuda \
  --no-save-results
```

Output atteso:
- `results/JETSON_SMOKE/benchmark.json`

---

## B) Docker (opzionale, se vuoi isolamento)

### B1) Installa Docker

Su Jetson (Ubuntu), una base comune è:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
docker version
```

### B2) Abilita GPU in Docker (NVIDIA runtime)

Con JetPack spesso è già disponibile il supporto. Verifica che `docker run --gpus all` funzioni. Se non funziona, devi installare/configurare NVIDIA Container Toolkit per Jetson (dipende dalla tua release).

### B3) Scegli l’immagine base

Il [jetson/Dockerfile](jetson/Dockerfile) richiede una base con PyTorch+CUDA per Jetson (es. una immagine NVIDIA L4T PyTorch) coerente con il tuo JetPack.

Esempio (placeholder):

```bash
export BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:<TAG_COMPATIBILE_CON_JETPACK>
```

### B4) Build dell’immagine

Il build va lanciato dalla root del repo (`BotSort-vs-TrackSSM-on-Nuscenes/`), perché il Dockerfile fa `COPY tesi_project_amarino/`.

```bash
cd BotSort-vs-TrackSSM-on-Nuscenes
docker build -f jetson/Dockerfile \
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  -t track-bench:jetson .
```

### B5) Run del container montando dataset e pesi

È meglio montare i path invece di “copiare” dataset/pesi nell’immagine.

```bash
cd BotSort-vs-TrackSSM-on-Nuscenes
docker run --rm -it \
  --runtime nvidia \
  --network host \
  -v $PWD/tesi_project_amarino/data:/workspace/tesi_project_amarino/data \
  -v $PWD/tesi_project_amarino/weights:/workspace/tesi_project_amarino/weights \
  -v $PWD/tesi_project_amarino/yolox_finetuning:/workspace/tesi_project_amarino/yolox_finetuning \
  track-bench:jetson
```

Dentro al container:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 track.py --help
```

Poi lancia lo smoke test come nella sezione A9.
```

## 7) Config “fast” vs “best metrics”

## 7) Config “fast” vs “best metrics”

- **FAST (più FPS)**: `--cmc-method none --trackssm-batch`
- **BEST METRICS**: `--cmc-method sparseOptFlow --trackssm-batch`

Il batching (`--trackssm-batch`) è la principale ottimizzazione implementata: riduce drasticamente la latenza del tracker evitando forward separati per ogni track.

## 8) Note utili

- Per timing realistici su GPU: usa `--benchmark-sync-cuda`.
- Per misurare solo compute: lascia `--benchmark-include-io` disattivo (default).
- Evita I/O: `--no-save-results` e niente `--save-videos`.

(Guida container opzionale: vedi `docs/JETSON_TESTING.md`.)
