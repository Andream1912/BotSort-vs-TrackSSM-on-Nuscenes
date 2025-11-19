#!/usr/bin/env python
"""
Test rapido del pipeline di training completo
Verifica che tutti i componenti funzionino prima del training notturno
"""

import sys
import os
import yaml
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_dataset_exists(data_root):
    """Verifica che il dataset esista"""
    print("\n" + "="*80)
    print("TEST 1: Dataset Existence")
    print("="*80)
    
    data_root = Path(data_root)
    
    checks = {
        'train': data_root / 'train',
        'val': data_root / 'val',
        'test': data_root / 'test'
    }
    
    for split, path in checks.items():
        if path.exists():
            seq_count = len(list(path.iterdir()))
            print(f"✅ {split}: {path} ({seq_count} sequences)")
        else:
            print(f"❌ {split}: {path} NOT FOUND")
            return False
    
    return True


def test_dataset_loader(data_root, device='cpu'):
    """Test il dataset loader"""
    print("\n" + "="*80)
    print("TEST 2: Dataset Loader")
    print("="*80)
    
    try:
        from dataset.nuscenes_interpolated_dataset import NuScenesInterpolatedDataset, collate_fn
        from torch.utils.data import DataLoader
        
        print("✅ Import NuScenesInterpolatedDataset")
        
        # Test train split
        dataset = NuScenesInterpolatedDataset(
            data_root=data_root,
            split='train',
            seq_len=20,
            max_objects=300
        )
        
        print(f"✅ Dataset train loaded: {len(dataset)} samples")
        
        # Test dataloader
        loader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=0,  # No workers for test
            collate_fn=collate_fn
        )
        
        print(f"✅ DataLoader created: {len(loader)} batches")
        
        # Test one batch
        batch = next(iter(loader))
        print(f"✅ Batch loaded successfully")
        print(f"   - condition: {batch['condition'].shape}")
        print(f"   - cur_bbox: {batch['cur_bbox'].shape}")
        print(f"   - track_ids: {batch['track_ids'].shape}")
        print(f"   - class_ids: {batch['class_ids'].shape}")
        print(f"   - padding_mask: {batch['padding_mask'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loader failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(config_path, device='cpu'):
    """Test caricamento modelli"""
    print("\n" + "="*80)
    print("TEST 3: Model Loading")
    print("="*80)
    
    try:
        # Import models
        from models.d2mp import D2MP
        from models.mamba_encoder import Mamba_encoder
        
        print("✅ Import models")
        
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        print(f"✅ Config loaded: {config_path}")
        
        # Create encoder
        encoder = Mamba_encoder(
            d_model=config_dict.get('d_model', 256),
            n_layer=config_dict.get('n_layer', 8),
            vocab_size=1
        ).to(device)
        
        print(f"✅ Encoder created: {sum(p.numel() for p in encoder.parameters()):,} params")
        
        # Create decoder
        decoder = D2MP(
            d_model=config_dict.get('d_model', 256),
            num_decoder_layers=config_dict.get('num_decoder_layers', 6),
            num_classes=config_dict.get('num_classes', 7)
        ).to(device)
        
        print(f"✅ Decoder created: {sum(p.numel() for p in decoder.parameters()):,} params")
        
        # Freeze encoder (Phase 1)
        for param in encoder.parameters():
            param.requires_grad = False
        
        trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
        
        print(f"✅ Phase 1 freeze applied:")
        print(f"   - Trainable (decoder): {trainable:,}")
        print(f"   - Frozen (encoder): {frozen:,}")
        
        return encoder, decoder
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(encoder, decoder, data_root, device='cpu'):
    """Test forward pass completo"""
    print("\n" + "="*80)
    print("TEST 4: Forward Pass")
    print("="*80)
    
    try:
        from dataset.nuscenes_interpolated_dataset import NuScenesInterpolatedDataset, collate_fn
        from torch.utils.data import DataLoader
        
        # Load one batch
        dataset = NuScenesInterpolatedDataset(
            data_root=data_root,
            split='train',
            seq_len=20,
            max_objects=300
        )
        
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
        batch = next(iter(loader))
        
        # Move to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        
        print("✅ Batch moved to device")
        
        # Create combined model
        class CombinedModel(torch.nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
            
            def forward(self, batch):
                return self.decoder(batch)
        
        model = CombinedModel(encoder, decoder).to(device)
        model.train()
        
        print("✅ Combined model created")
        
        # Forward pass
        with torch.set_grad_enabled(True):
            loss_dict = model(batch)
        
        print("✅ Forward pass successful")
        print(f"   Loss dict keys: {list(loss_dict.keys())}")
        
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                print(f"   - {k}: {v.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(encoder, decoder, data_root, device='cpu'):
    """Test backward pass e optimizer"""
    print("\n" + "="*80)
    print("TEST 5: Backward Pass & Optimizer")
    print("="*80)
    
    try:
        from dataset.nuscenes_interpolated_dataset import NuScenesInterpolatedDataset, collate_fn
        from torch.utils.data import DataLoader
        
        # Load one batch
        dataset = NuScenesInterpolatedDataset(
            data_root=data_root,
            split='train',
            seq_len=20,
            max_objects=300
        )
        
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
        batch = next(iter(loader))
        
        # Move to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        
        # Create combined model
        class CombinedModel(torch.nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
            
            def forward(self, batch):
                return self.decoder(batch)
        
        model = CombinedModel(encoder, decoder).to(device)
        model.train()
        
        # Create optimizer (only decoder params)
        optimizer = torch.optim.AdamW(
            [p for p in model.decoder.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=0.01
        )
        
        print("✅ Optimizer created")
        
        # Forward
        loss_dict = model(batch)
        
        # Check loss
        if 'loss' in loss_dict:
            total_loss = loss_dict['loss']
        else:
            total_loss = sum(v for k, v in loss_dict.items() if 'loss' in k.lower() and torch.is_tensor(v))
        
        print(f"✅ Total loss computed: {total_loss.item():.4f}")
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        print("✅ Backward pass successful")
        
        # Check gradients
        grad_norms = []
        for name, param in model.decoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            avg_grad = sum(grad_norms) / len(grad_norms)
            max_grad = max(grad_norms)
            print(f"✅ Gradients computed:")
            print(f"   - Avg grad norm: {avg_grad:.6f}")
            print(f"   - Max grad norm: {max_grad:.6f}")
        else:
            print("❌ No gradients found!")
            return False
        
        # Optimizer step
        optimizer.step()
        
        print("✅ Optimizer step successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("TRAINING PIPELINE TEST")
    print("="*80)
    print("Questo test verifica tutti i componenti prima del training notturno")
    print()
    
    # Parametri
    config_path = 'configs/nuscenes_phase1.yaml'
    data_root = '/mnt/datasets/Nuscense/nuscenes_mot_6cams_interpolated'
    device = 'cpu'  # Test sempre su CPU per sicurezza
    
    print(f"Config: {config_path}")
    print(f"Data root: {data_root}")
    print(f"Device: {device}")
    
    results = []
    
    # Test 1: Dataset exists
    results.append(("Dataset Existence", test_dataset_exists(data_root)))
    
    if not results[-1][1]:
        print("\n❌ Dataset non trovato. Attendi che la generazione finisca.")
        return False
    
    # Test 2: Dataset loader
    results.append(("Dataset Loader", test_dataset_loader(data_root, device)))
    
    if not results[-1][1]:
        print("\n❌ Dataset loader fallito.")
        return False
    
    # Test 3: Model loading
    encoder, decoder = test_model_loading(config_path, device)
    results.append(("Model Loading", encoder is not None and decoder is not None))
    
    if not results[-1][1]:
        print("\n❌ Model loading fallito.")
        return False
    
    # Test 4: Forward pass
    results.append(("Forward Pass", test_forward_pass(encoder, decoder, data_root, device)))
    
    if not results[-1][1]:
        print("\n❌ Forward pass fallito.")
        return False
    
    # Test 5: Backward pass
    results.append(("Backward Pass", test_backward_pass(encoder, decoder, data_root, device)))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 TUTTI I TEST PASSATI!")
        print("   Il training può partire in sicurezza per la notte.")
        print("="*80)
        return True
    else:
        print("\n❌ ALCUNI TEST FALLITI")
        print("   Controlla gli errori sopra prima di avviare il training.")
        print("="*80)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
