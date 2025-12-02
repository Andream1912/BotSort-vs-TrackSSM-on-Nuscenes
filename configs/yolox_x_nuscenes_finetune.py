#!/usr/bin/env python3
"""
YOLOX Fine-tuning Configuration for NuScenes
Adapted from YOLOX default config
"""
import os
from yolox.exp import Exp as MyExp
import torch
import torch.distributed as dist

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # ============ Model Config ============
        self.num_classes = 7  # car, pedestrian, bicycle, motorcycle, bus, trailer, truck
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # ============ Transform Config ============
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.5
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        
        # ============ Training Config ============
        self.warmup_epochs = 1
        self.max_epoch = 20
        self.warmup_lr = 0.00001
        self.basic_lr_per_img = 0.0001 / 64.0  # Reduced LR for fine-tuning
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 2
        self.min_lr_ratio = 0.05
        self.ema = True
        
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 5  # Evaluate every 5 epochs
        self.save_history_ckpt = True
        
        # ============ Data Loader Config ============
        self.data_num_workers = 4
        self.input_size = (896, 1600)  # NuScenes camera resolution
        self.multiscale_range = 5
        self.random_size = (16, 26)
        self.test_size = (896, 1600)
        
        # ============ Dataset Config ============
        self.data_dir = "data/nuscenes_yolox_detector"
        self.train_ann = "annotations/train.json"
        self.val_ann = "annotations/val.json"
        
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """Override to use COCO dataset loader"""
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
                json_file=self.train_ann,
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
        
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        
        return train_loader
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Override to use validation set"""
        from yolox.data import COCODataset, ValTransform
        
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val2017",
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
        """Override to use COCO evaluator"""
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
