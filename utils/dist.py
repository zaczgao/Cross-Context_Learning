import os
import sys
import random
import datetime

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    if is_dist_avail_and_initialized():
        return
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(random.randint(0, 9999) + 40000)
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    print("Use GPU: {} ranked {} out of {} gpus for training".format(args.gpu, args.rank, args.world_size))
    if args.multiprocessing_distributed:
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            timeout=datetime.timedelta(hours=2),
            rank=args.rank,
        )
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        dist.barrier()

    torch.cuda.set_device(args.gpu)
    setup_for_distributed(args.rank == 0)


def cleanup_multigpu():
	dist.destroy_process_group()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def all_reduce_mean(x):
    # reduce tensore for DDP
    # source: https://raw.githubusercontent.com/NVIDIA/apex/master/examples/imagenet/main_amp.py
    world_size = get_world_size()
    if world_size > 1:
        rt = x.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= world_size
        return rt
    else:
        return x


def reduce_loss_dict(loss_pack):
	with torch.no_grad():
		keys = []
		losses = []
		for key, value in loss_pack.items():
			keys.append(key)
			losses.append(value)

		losses = torch.stack(losses, 0)
		losses = all_reduce_mean(losses)

		loss_reduce = {k: v for k, v in zip(keys, losses)}

	return loss_reduce


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
