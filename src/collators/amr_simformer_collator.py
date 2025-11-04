import torch
from torch_geometric.data import Batch
from torch_geometric.nn.pool import radius_graph
from kappadata.collators import CollatorBase


class AMRSimformerCollator(CollatorBase):
    """
    Collator for AMR dataset with Simformer model.
    Similar to LagrangianSimformerCollator.
    """
    
    def __init__(
        self,
        graph_mode='radius_graph_with_supernodes',
        radius_graph_r=0.05,
        radius_graph_max_num_neighbors=10000,
        n_supernodes=128,
        supernode_mode='random',
        knn_graph_k=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph_mode = graph_mode
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors
        self.n_supernodes = n_supernodes
        self.supernode_mode = supernode_mode
        self.knn_graph_k = knn_graph_k
    
    def collate(self, items):
        """
        Collate batch of AMR samples.
        
        Args:
            items: List of dicts with keys:
                - curr_pos: (N, 2) positions
                - x: (N, D) input features
                - target: (N, 6) target fields
                - timestep: (1,) timestep index
        
        Returns:
            batch: Dict with batched tensors and graph structure
        """
        # Stack all samples into single batch
        batch_size = len(items)
        
        # Collect positions and features
        all_pos = []
        all_x = []
        all_target = []
        all_timesteps = []
        batch_indices = []
        
        for i, item in enumerate(items):
            n_points = item['curr_pos'].shape[0]
            all_pos.append(item['curr_pos'])
            all_x.append(item['x'])
            all_target.append(item['target'])
            all_timesteps.append(item['timestep'])
            batch_indices.append(torch.full((n_points,), i, dtype=torch.long))
        
        # Concatenate
        curr_pos = torch.cat(all_pos, dim=0)  # (total_N, 2)
        x = torch.cat(all_x, dim=0)  # (total_N, D)
        target = torch.cat(all_target, dim=0)  # (total_N, 6)
        batch_idx = torch.cat(batch_indices, dim=0)  # (total_N,)
        timestep = torch.cat(all_timesteps, dim=0)  # (batch_size,)
        
        # Build graphs
        if 'radius_graph' in self.graph_mode:
            # Build radius graph per sample in batch
            edge_index_list = []
            offset = 0
            
            for i in range(batch_size):
                mask = (batch_idx == i)
                pos_i = curr_pos[mask]
                
                # Build radius graph for this sample
                edge_index_i = radius_graph(
                    pos_i,
                    r=self.radius_graph_r,
                    max_num_neighbors=self.radius_graph_max_num_neighbors,
                    loop=False,
                )
                
                # Add offset
                edge_index_i = edge_index_i + offset
                edge_index_list.append(edge_index_i)
                offset += pos_i.shape[0]
            
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            raise NotImplementedError(f"Graph mode {self.graph_mode} not implemented")
        
        # Handle supernodes if needed
        if 'supernodes' in self.graph_mode:
            # Create supernodes for each sample
            supernode_pos_list = []
            edge_index_target_list = []
            
            offset = 0
            offset_super = 0
            
            for i in range(batch_size):
                mask = (batch_idx == i)
                pos_i = curr_pos[mask]
                n_points = pos_i.shape[0]
                
                # Sample supernodes
                if self.supernode_mode == 'random':
                    supernode_indices = torch.randperm(n_points)[:self.n_supernodes]
                    supernode_pos = pos_i[supernode_indices]
                else:
                    raise NotImplementedError(f"Supernode mode {self.supernode_mode} not implemented")
                
                supernode_pos_list.append(supernode_pos)
                
                # Connect each cell to nearest supernodes
                edge_index_to_super = radius_graph(
                    pos_i,
                    r=self.radius_graph_r * 2,  # Larger radius for supernodes
                    max_num_neighbors=self.n_supernodes,
                    loop=False,
                )
                
                # Add offsets
                edge_index_to_super[0] += offset
                edge_index_to_super[1] += offset_super
                edge_index_target_list.append(edge_index_to_super)
                
                offset += n_points
                offset_super += self.n_supernodes
            
            # Concatenate supernodes
            curr_pos_full = torch.cat([curr_pos] + supernode_pos_list, dim=0)
            edge_index_target = torch.cat(edge_index_target_list, dim=1)
        else:
            curr_pos_full = curr_pos
            edge_index_target = None
        
        return {
            'x': x,
            'curr_pos': curr_pos,
            'curr_pos_full': curr_pos_full,
            'edge_index': edge_index,
            'edge_index_target': edge_index_target,
            'timestep': timestep,
            'target': target,
            'batch': batch_idx,
        }

