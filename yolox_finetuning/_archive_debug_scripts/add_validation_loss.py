#!/usr/bin/env python3
"""
Script per aggiungere il calcolo della validation loss a YOLOX trainer
"""

import sys
import os

# Path al file trainer.py
TRAINER_FILE = "/user/amarino/tesi_project_amarino/external/YOLOX/yolox/core/trainer.py"

# Codice da aggiungere per calcolare validation loss
VALIDATION_LOSS_CODE = '''
    def compute_validation_loss(self):
        """Compute validation loss on the validation dataset"""
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        evalmodel.eval()
        
        # Get validation dataloader
        val_loader = self.exp.get_eval_loader(
            batch_size=self.exp.eval_batch_size if hasattr(self.exp, 'eval_batch_size') else self.args.batch_size,
            is_distributed=self.is_distributed,
            testdev=False,
            legacy=False
        )
        
        total_loss = 0.0
        num_batches = 0
        
        logger.info("Computing validation loss...")
        
        with torch.no_grad():
            for batch_idx, (imgs, targets, _, _) in enumerate(val_loader):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = evalmodel(imgs, targets=targets)
                loss = outputs["total_loss"]
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log progress every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Validation: [{batch_idx + 1}/{len(val_loader)}] batches processed")
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss
'''

# Codice modificato per evaluate_and_save_model che include validation loss
MODIFIED_EVALUATE_FUNCTION = '''    def evaluate_and_save_model(self):
        # Compute validation loss first
        val_loss = self.compute_validation_loss()
        
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
                self.tblogger.add_scalar("val/loss", val_loss, self.epoch + 1)  # Add validation loss
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics(
                    {
                        "val/COCOAP50": ap50,
                        "val/COCOAP50_95": ap50_95,
                        "val/loss": val_loss,  # Add validation loss
                        "train/epoch": self.epoch + 1,
                    }
                )
                self.wandb_logger.log_images(predictions)
            logger.info("\\n" + summary)
            logger.info(f"Validation Loss: {val_loss:.4f}")
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)
'''

def modify_trainer():
    """Modify the trainer.py file to add validation loss computation"""
    
    # Backup del file originale
    backup_file = TRAINER_FILE + ".backup_before_valloss"
    if not os.path.exists(backup_file):
        os.system(f"cp {TRAINER_FILE} {backup_file}")
        print(f"✅ Backup creato: {backup_file}")
    
    with open(TRAINER_FILE, 'r') as f:
        content = f.read()
    
    # Check se già modificato
    if "compute_validation_loss" in content:
        print("⚠️  Il file è già stato modificato. Nessuna modifica necessaria.")
        return
    
    # Trova dove inserire compute_validation_loss (prima di evaluate_and_save_model)
    marker = "    def evaluate_and_save_model(self):"
    if marker not in content:
        print("❌ Errore: non trovata la funzione evaluate_and_save_model")
        return
    
    # Inserisci compute_validation_loss prima di evaluate_and_save_model
    content = content.replace(marker, VALIDATION_LOSS_CODE + "\n" + marker)
    
    # Sostituisci evaluate_and_save_model con la versione modificata
    # Trova l'inizio e la fine della funzione originale
    start_idx = content.find("    def evaluate_and_save_model(self):")
    if start_idx == -1:
        print("❌ Errore: funzione evaluate_and_save_model non trovata")
        return
    
    # Trova la prossima funzione (def save_ckpt)
    end_marker = "    def save_ckpt(self, ckpt_name"
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        print("❌ Errore: non trovata save_ckpt dopo evaluate_and_save_model")
        return
    
    # Estrai la parte da sostituire
    original_function = content[start_idx:end_idx]
    
    # Sostituisci con la nuova versione
    content = content.replace(original_function, MODIFIED_EVALUATE_FUNCTION + "\n")
    
    # Scrivi il file modificato
    with open(TRAINER_FILE, 'w') as f:
        f.write(content)
    
    print("✅ File trainer.py modificato con successo!")
    print("✅ Ora il training calcolerà anche la validation loss")
    print(f"✅ Backup salvato in: {backup_file}")

if __name__ == "__main__":
    modify_trainer()
