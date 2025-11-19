"""
Callback to compute and log relative L1 and L2 errors.
"""
import torch
from callbacks.base.periodic_callback import PeriodicCallback


class RelativeErrorCallback(PeriodicCallback):
    """
    Compute relative L1 and L2 errors for model predictions.
    
    Relative L1: sum(|pred - target|) / sum(|target|)
    Relative L2: ||pred - target||_2 / ||target||_2
    """
    
    def __init__(
        self,
        dataset_key="train",
        forward_kwargs=None,
        log_key_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.forward_kwargs = forward_kwargs or {}
        self.log_key_prefix = log_key_prefix or f"rel_error/{dataset_key}"
    
    def _periodic_callback(self, model, **_):
        """Compute relative errors on the dataset."""
        # Get dataset and dataloader
        ds, collator = self.data_container.get_dataset(self.dataset_key, mode="x")
        loader = self.data_container.get_dataloader(
            self.dataset_key,
            batch_size=1,
            shuffle=False,
        )
        
        model.eval()
        rel_l1_sum = 0.0
        rel_l2_sum = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                # Unpack batch
                batch_tuple, batch_ctx = batch
                
                # Prepare inputs
                inputs = {
                    "x": batch_tuple[0].to(self.device),
                    "mesh_pos": batch_tuple[1].to(self.device),
                    "query_pos": batch_tuple[2].to(self.device),
                    "mesh_edges": batch_tuple[3].to(self.device) if batch_tuple[3] is not None else None,
                    "geometry2d": batch_tuple[4].to(self.device),
                    "timestep": batch_tuple[5].to(self.device),
                    "velocity": batch_tuple[6].to(self.device),
                    "batch_idx": batch_ctx["batch_idx"].to(self.device),
                    "unbatch_idx": batch_ctx["unbatch_idx"].to(self.device),
                    "unbatch_select": batch_ctx["unbatch_select"].to(self.device),
                }
                target = batch_tuple[7].to(self.device)
                
                # Forward pass
                outputs = model(**inputs, **self.forward_kwargs)
                pred = outputs.get("x_hat")
                
                if pred is None:
                    self.logger.warning("No 'x_hat' in model outputs, skipping relative error computation")
                    continue
                
                # Compute relative errors
                eps = 1e-12
                
                # Relative L1
                rel_l1 = (torch.abs(pred - target).sum() / (torch.abs(target).sum() + eps)).item()
                
                # Relative L2
                pred_flat = pred.reshape(pred.shape[0], -1)
                target_flat = target.reshape(target.shape[0], -1)
                rel_l2 = (
                    torch.linalg.vector_norm(pred_flat - target_flat, ord=2, dim=1).sum() / 
                    (torch.linalg.vector_norm(target_flat, ord=2, dim=1).sum() + eps)
                ).item()
                
                rel_l1_sum += rel_l1
                rel_l2_sum += rel_l2
                n_samples += 1
        
        # Log averages
        if n_samples > 0:
            avg_rel_l1 = rel_l1_sum / n_samples
            avg_rel_l2 = rel_l2_sum / n_samples
            
            self.writer.add_scalar(
                f"{self.log_key_prefix}/rel_l1",
                avg_rel_l1,
                self.update_counter.cur_epoch,
            )
            self.writer.add_scalar(
                f"{self.log_key_prefix}/rel_l2",
                avg_rel_l2,
                self.update_counter.cur_epoch,
            )
            
            self.logger.info(
                f"{self.dataset_key} rel_l1={avg_rel_l1:.6f} rel_l2={avg_rel_l2:.6f}"
            )
        
        model.train()



