#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Model
        self.depth = 1.0
        self.width = 1.0
        self.num_classes = 7
        
        # Training settings - STANDARD YOLOX
        self.max_epoch = 10
        self.warmup_epochs = 1
        self.no_aug_epochs = 2
        
        # Data settings
        self.data_num_workers = 0  # Single thread to avoid index conflicts
        # Match NuScenes aspect ratio (900x1600) with higher resolution
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (14, 20)  # Keep aspect ratio during multi-scale
        
        # Batch and learning rate - HIGHER LR for head-only training
        self.batch_size = 8
        self.basic_lr_per_img = 0.001 / 64.0  # Lower LR (1x) to compare
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        
        # Data augmentation - REDUCED
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.5
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 5.0
        self.translate = 0.1
        self.mosaic_scale = (0.8, 1.2)
        self.mixup_scale = (0.8, 1.2)
        self.shear = 1.0
        self.enable_mixup = True
        
        # Optimization
        self.momentum = 0.9
        self.weight_decay = 5e-4
        
        # Eval
        self.test_conf = 0.01
        self.nmsthre = 0.65
        
        # Output
        self.output_dir = "./yolox_finetuning"
        self.exp_name = "yolox_l_nuscenes_clean_v2"
        
        # Data paths - Use YOLOX format dataset
        self.data_dir = "data/nuscenes_yolox_detector"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        
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
        
        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann if not no_aug else self.val_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob
                ),
                cache=cache_img,
            )
        
        if not no_aug:
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
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
        
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
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
