#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# YOLOX-X Fine-tuning on nuScenes (Optimized)

import os
from yolox.exp import Exp as MyExp
import torch
import torch.distributed as dist

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Model
        self.depth = 1.33
        self.width = 1.25
        self.num_classes = 7  # nuScenes classes
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Input size
        self.input_size = (896, 1600)  # (H, W) - multiple of 32
        self.test_size = (896, 1600)
        self.random_size = (18, 32)  # Random resize range
        
        # Data augmentation
        self.mosaic_prob = 0.5  # Reduce for fine-tuning
        self.mixup_prob = 0.3   # Reduce for fine-tuning
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 5.0      # Reduce rotation for vehicles
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 1.0
        self.enable_mixup = True
        
        # Training settings - OPTIMIZED
        self.max_epoch = 30
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.001 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 5  # Last 5 epochs without aug
        self.min_lr_ratio = 0.05
        self.ema = True
        
        # Batch settings - OPTIMIZED
        self.data_num_workers = 8  # Parallel data loading
        self.input_size = (896, 1600)
        self.multiscale_range = 5
        
        # Mixed precision - 2x SPEEDUP
        self.enable_amp = True  # Enable automatic mixed precision
        
        # Evaluation - DISABLED during training
        self.eval_interval = 100  # Only eval at end
        
        # Save checkpoints
        self.save_history_ckpt = True
        self.ckpt_interval = 5  # Save every 5 epochs
        
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master
        
        dataset = COCODataset(
            data_dir="data/nuscenes_yolox_detector",
            json_file="train.json",  # FIXED: removed 'annotations/' prefix
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache_img,
        )
        
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        
        self.dataset = dataset
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }
        
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        
        return train_loader
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform
        
        valdataset = COCODataset(
            data_dir="data/nuscenes_yolox_detector",
            json_file="val.json",  # FIXED: removed 'annotations/' prefix
            name="val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        
        return val_loader
    
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator
        
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=0.01,
            nmsthre=0.65,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
