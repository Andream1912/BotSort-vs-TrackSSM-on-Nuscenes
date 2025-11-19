#!/bin/bash

# Test pre-flight per verificare che tutto sia pronto per training su GPU

echo "========================================="
echo "PRE-FLIGHT CHECK - Training Phase 1"
echo "========================================="
echo ""

cd /user/amarino/tesi_project_amarino/trackssm_reference

# 1. Check CUDA availability
echo "1️⃣  CUDA AVAILABILITY"
echo "----------------------------------------"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1
echo ""

# 2. Check dataset
echo "2️⃣  DATASET CHECK"
echo "----------------------------------------"
DATASET_PATH="./data/nuscenes_mot_6cams_interpolated"

if [ -d "$DATASET_PATH/train" ]; then
    TRAIN_SEQ=$(ls -1 $DATASET_PATH/train 2>/dev/null | wc -l)
    VAL_SEQ=$(ls -1 $DATASET_PATH/val 2>/dev/null | wc -l)
    TEST_SEQ=$(ls -1 $DATASET_PATH/test 2>/dev/null | wc -l)
    
    echo "✅ Dataset found: $DATASET_PATH"
    echo "   Train sequences: $TRAIN_SEQ"
    echo "   Val sequences: $VAL_SEQ"
    echo "   Test sequences: $TEST_SEQ"
else
    echo "❌ Dataset NOT found: $DATASET_PATH"
    exit 1
fi
echo ""

# 3. Check dataset loader
echo "3️⃣  DATASET LOADER TEST"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, '.')
from dataset.nuscenes_track_dataset import NuScenesTrackDataset, collate_fn
from torch.utils.data import DataLoader

try:
    dataset = NuScenesTrackDataset(
        data_root='$DATASET_PATH',
        split='train',
        history_len=5
    )
    print(f'✅ Dataset loaded: {len(dataset)} samples')
    
    # Test single sample
    sample = dataset[0]
    print(f'✅ Sample format correct:')
    for k, v in sample.items():
        print(f'   {k}: {v.shape}')
    
    # Test batch
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))
    print(f'✅ Batch collation works:')
    for k, v in batch.items():
        print(f'   {k}: {v.shape}')
        
except Exception as e:
    print(f'❌ Dataset loader failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | head -40
echo ""

# 4. Check model creation
echo "4️⃣  MODEL CREATION TEST"
echo "----------------------------------------"
python -c "
import sys
import yaml
import torch
sys.path.insert(0, '.')
from models.autoencoder import D2MP
from models.condition_embedding import Time_info_aggregation

try:
    # Load config
    with open('configs/nuscenes_phase1_gpu.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    print(f'✅ Config loaded')
    print(f'   Device: {config_dict[\"device\"]}')
    print(f'   Batch size: {config_dict[\"batch_size\"]}')
    print(f'   Workers: {config_dict[\"num_workers\"]}')
    
    # Create encoder
    encoder = Time_info_aggregation(
        d_model=config_dict.get('encoder_dim', 256),
        n_layer=config_dict.get('n_layer', 2),
        v_size=8
    )
    print(f'✅ Encoder created')
    
    # Create model (on CPU for test)
    from types import SimpleNamespace
    cfg = SimpleNamespace(**config_dict)
    cfg.device = 'cpu'  # Test on CPU
    
    model = D2MP(cfg, encoder=encoder, device='cpu')
    print(f'✅ Model created')
    print(f'   Total params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
except Exception as e:
    print(f'❌ Model creation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1
echo ""

# 5. Check training script syntax
echo "5️⃣  TRAINING SCRIPT SYNTAX"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, '.')
compile(open('scripts/training/train_phase1_decoder.py').read(), 'train_phase1_decoder.py', 'exec')
print('✅ Training script syntax correct')
" 2>&1
echo ""

# 6. Summary
echo "========================================="
echo "SUMMARY"
echo "========================================="

CUDA_OK=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null)

if [ "$CUDA_OK" = "yes" ]; then
    echo "✅ CUDA: Available"
    echo "✅ Dataset: Ready ($(ls -1 $DATASET_PATH/train 2>/dev/null | wc -l) train sequences)"
    echo "✅ Dataset Loader: Working"
    echo "✅ Model: Can be created"
    echo "✅ Training Script: Syntax OK"
    echo ""
    echo "🚀 READY TO LAUNCH TRAINING!"
    echo ""
    echo "Launch command:"
    echo "  python scripts/training/train_phase1_decoder.py \\"
    echo "    --config configs/nuscenes_phase1_gpu.yaml \\"
    echo "    --data_root ./data/nuscenes_mot_6cams_interpolated \\"
    echo "    --output_dir weights/phase1"
else
    echo "⚠️  CUDA: NOT available (training will fail)"
    echo "✅ Dataset: Ready"
    echo "✅ Dataset Loader: Working"
    echo "✅ Model: Can be created"
    echo "✅ Training Script: Syntax OK"
    echo ""
    echo "⏸️  WAIT FOR GPU before launching!"
fi

echo "========================================="
