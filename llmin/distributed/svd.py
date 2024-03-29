import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import os

WORLD_SIZE = torch.cuda.device_count()

def setup(rank, world_size = WORLD_SIZE):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Assuming your data is loaded and available as `data_tensor`
def distribute_data(data_tensor, rank, world_size = WORLD_SIZE):
    # Split data among the available GPUs
    chunk_size = data_tensor.size(0) // world_size
    chunks = data_tensor.chunk(world_size, dim=0)
    distributed_chunk = torch.empty_like(chunks[0])
    dist.scatter(distributed_chunk, chunks if rank == 0 else None, src=0)
    return distributed_chunk

def compute_svd(rank, data_chunk):
    U, S, V = torch.linalg.svd(data_chunk, full_matrices=False)
    gathered_S = [torch.zeros_like(S) for _ in range(WORLD_SIZE)]
    dist.gather(S, gather_list=gathered_S if rank == 0 else None, dst=0)
    if rank == 0:
        # Process or save gathered_S
        pass