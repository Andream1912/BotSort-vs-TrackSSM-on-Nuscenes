#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import math
import torch
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Model
        self.depth = 1.0
        self.width = 1.0
        self.num_classes = 7
        
        # Training settings - ULTRA STABLE V3
        self.max_epoch = 30  # Extended training for better convergence
        self.warmup_epochs = 3  # Warmup più lungo (3 epoche) per stabilizzazione iniziale
        self.no_aug_epochs = 8  # Ultime 8 epoche senza aug (epoch 22-30) per fine-tune smooth
        
        # Data settings
        self.data_num_workers = 0  # Single thread to avoid index conflicts
        # Match NuScenes aspect ratio (900x1600)
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (14, 20)  # Keep aspect ratio during multi-scale
        
        # Batch and learning rate - ULTRA STABLE
        self.batch_size = 32  # Large batch for gradient stability
        self.basic_lr_per_img = 0.0006 / 64.0  # Ridotto 40% (era 0.001) per convergenza smooth
        self.warmup_lr = 0
        self.min_lr_ratio = 0.10  # Decay molto più graduale (era 0.01)
        
        # Data augmentation - MINIMAL per massima stabilità
        self.mosaic_prob = 0.2  # Ridotto ulteriormente da 0.3 -> 0.2
        self.mixup_prob = 0.15  # Ridotto molto da 0.3 -> 0.15
        self.hsv_prob = 0.4     # Ridotto leggermente per meno varianza colore
        self.flip_prob = 0.5    # OK, flip è stabile
        self.degrees = 3.0      # Ridotto da 5.0 -> rotazione più conservativa
        self.translate = 0.08   # Ridotto da 0.1 -> traslazione più conservativa
        self.mosaic_scale = (0.85, 1.15)  # Range più stretto
        self.mixup_scale = (0.85, 1.15)   # Range più stretto
        self.shear = 0.5        # Ridotto da 1.0 -> shear più conservativo
        self.enable_mixup = True
        
        # Optimization
        self.momentum = 0.9
        self.weight_decay = 5e-4
        
        # Scheduler - COSINE ANNEALING for smooth decay
        self.scheduler = "yoloxwarmcos"  # YOLOX's cosine annealing with warmup
        
        # Eval
        self.eval_interval = 1  # Run validation every epoch
        self.test_conf = 0.01
        self.nmsthre = 0.65
        
        # Output
        self.output_dir = "./yolox_finetuning"
        self.exp_name = "yolox_l_nuscenes_stable"  # V3: Ultra stable configuration
        
        # Data paths - ABSOLUTE PATHS
        self.data_dir = "/user/amarino/tesi_project_amarino/data/nuscenes_yolox_detector"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        
    def get_lr_scheduler(self, lr, iters_per_epoch):
        """
        Cosine annealing scheduler for smooth convergence
        """
        from yolox.utils import LRScheduler
        
        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler
        
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

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val2017",  # Use val2017 symlink for images
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
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
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
