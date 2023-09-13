import os
import sys
import numpy as np
import faiss
import copy

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet
from classy_vision.generic.distributed_util import is_distributed_training_run

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from configs.options import ClusterOptions
from utils.dist import init_distributed_mode, get_rank
from utils.util import MetricLogger, decode_imagenet
from utils.metric import evaluate_cluster


def run_hkmeans(x, num_clusters, base_temperature=0.2, local_rank=0, niters=20, nredos=5):
    """
    This function is a hierarchical
    k-means: the centroids of current hierarchy is used
    to perform k-means in next step.
    """

    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': [], 'cluster2cluster': [], 'logits': []}

    for seed, num_cluster in enumerate(num_clusters):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = niters
        clus.nredo = nredos
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        if sys.platform == 'win32':
            index = faiss.IndexFlatL2(d)
        else:
            if is_distributed_training_run():
                ngpu = torch.cuda.device_count()
            else:
                npu = 1

            res = [faiss.StandardGpuResources() for i in range(ngpu)]

            flat_config = []
            for i in range(ngpu):
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = i
                flat_config.append(cfg)

            if ngpu == 1:
                index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
            else:
                indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                           for i in range(ngpu)]
                index = faiss.IndexReplicas()
                for sub_index in indexes:
                    index.addIndex(sub_index)

        if seed == 0:  # the first hierarchy from instance directly
            clus.train(x, index)
            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        else:
            # the input of higher hierarchy is the centorid of lower one
            clus.train(results['centroids'][seed - 1].cpu().numpy(), index)
            D, I = index.search(results['centroids'][seed - 1].cpu().numpy(), 1)

        im2cluster = [int(n[0]) for n in I]
        # sample-to-centroid distances for each cluster
        ## centroid in lower level to higher level
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        if seed > 0:  # the im2cluster of higher hierarchy is the index of previous hierachy
            im2cluster = np.array(im2cluster)  # enable batch indexing
            results['cluster2cluster'].append(torch.LongTensor(im2cluster).cuda())
            im2cluster = im2cluster[results['im2cluster'][seed - 1].cpu().numpy()]
            im2cluster = list(im2cluster)

        if len(set(im2cluster)) == 1:
            print("Warning! All samples are assigned to one cluster")

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
        density = base_temperature * density / density.mean()

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)
        if seed > 0:  # maintain a logits from lower prototypes to higher
            proto_logits = torch.mm(results['centroids'][-1], centroids.t())
            results['logits'].append(proto_logits.cuda())

        density = torch.Tensor(density).cuda()
        im2cluster = torch.LongTensor(im2cluster).cuda()
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, label = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


def run_clustering(features, args):
    """
    eval clustering result by kmeans clustering with features and labels.

    Args:
        features: torch.tensor([N, D])
        labels: torch.tensor([N,])
    Returns:
        dict(acc, nmi, ami)
    """
    num_clusters = [int(x) for x in args.num_classes.split(",")]

    cluster_result = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster': [], 'logits': []}
    for i, num_cluster in enumerate(num_clusters):
        cluster_result['im2cluster'].append(torch.zeros(len(features),dtype=torch.long).cuda())
        cluster_result['centroids'].append(torch.zeros(int(num_cluster), features.shape[-1]).cuda())
        cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
        if i < (len(num_clusters) - 1):
            cluster_result['cluster2cluster'].append(torch.zeros(int(num_cluster), dtype=torch.long).cuda())
            cluster_result['logits'].append(torch.zeros([int(num_cluster), int(num_clusters[i+1])]).cuda())

    if dist.get_rank() == 0:
        features[torch.norm(features,dim=1)>1.5] /= 2 
        features = features.cpu().numpy()
        cluster_result = run_hkmeans(features, num_clusters) 

    dist.barrier()
    for k, data_list in cluster_result.items():
        for data_tensor in data_list:
            dist.broadcast(data_tensor, 0, async_op=False)
    
    im2cluster = cluster_result['im2cluster'][-1].cpu().long() # [N,]
    print("number of unique clusters: {}".format(len(im2cluster.unique())))
    print("max cluster id {}".format(im2cluster.max()))

    return im2cluster


@torch.no_grad()
def extract_features(model, loader, args):
    model.eval()
    header = 'Feature Extraction'
    log_interval = 100
    metric_logger = MetricLogger(delimiter="  ")
    ## lazy feat dim
    image = next(iter(loader))[0].cuda()
    feat = model(image)
    feat_dim = feat.shape[-1]

    features =  torch.zeros(len(loader.dataset), feat_dim).cuda()
    print("feature shape: {}".format(features.shape))

    for it, (image, index) in enumerate(metric_logger.log_every(loader, log_interval, header)):
        image = image.cuda(non_blocking=True)
        feat = F.normalize(model(image), dim=-1)
        features[index] = feat.detach()
    dist.barrier()
    dist.all_reduce(features, op=dist.ReduceOp.SUM)
    features[torch.norm(features,dim=1)>1.5] /= 2 
    return features.cpu()

def run(args):
    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_val = ReturnIndexDataset(os.path.join(args.data, "val"), transform=transform)
    test_sampler = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=10,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_val)} val imgs.")

    if args.arch in resnet.__dict__.keys():
        model = resnet.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    # elif args.arch in vt.__dict__.keys():
    #     if args.rel_pos_emb:
    #         model = vt.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, use_mean_pooling=False,
    #                                        use_abs_pos_emb=False, use_shared_rel_pos_bias=True)
    #     else:
    #         model = vt.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, use_mean_pooling=False)
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()

    def load_pretrained_weights(model, pretrained, backbone_prefix=None, model_prefix="model", filtered_keys=[]):
        checkpoint = torch.load(pretrained, map_location="cpu")
        if len(model_prefix):
            checkpoint_model = checkpoint[model_prefix]
        else:
            checkpoint_model = checkpoint
        ## automatically remove ddp prefix
        if all([k.startswith("module.") for k in checkpoint_model.keys()]):
            print("remove ddp prefix from model.")
            checkpoint_model = {k.replace("module.", ""):v for k,v in checkpoint_model.items()}
        
        if backbone_prefix:
            checkpoint_model = {k[len(backbone_prefix)+1:]:v for k,v in checkpoint_model.items() if k.startswith(backbone_prefix)}
        
        state_dict = model.state_dict()
        ## remove head / fc
        removed_keys = list()
        for key in checkpoint_model.keys():
            if key not in state_dict or key in filtered_keys or checkpoint_model[key].shape != state_dict[key].shape:
                removed_keys.append(key)

        print("removed keys in pretrained model: {}".format(removed_keys))
        for key in removed_keys:
            checkpoint_model.pop(key)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("loading message: {}".format(msg))
        return msg

    load_pretrained_weights(model, args.pretrained, args.backbone_prefix, args.model_prefix)
    model.eval()

    # ============ extract features... ============
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    test_features = extract_features(model, data_loader_val, args)

    if args.use_super_class:
        wordnet_is_a_txt_path = "./data/imagenet/wordnet.is_a.txt"
        words_txt_path = "./data/imagenet/words.txt"
        _, child_parent_idx = decode_imagenet(os.path.join(args.data, "val"), wordnet_is_a_txt_path, words_txt_path)
        test_labels_copy = copy.copy(test_labels)
        for child, parent in enumerate(child_parent_idx):
            test_labels[test_labels_copy == child] = parent

    im2cluster = run_clustering(test_features, args)
    nmi, ami, ari, f, acc = evaluate_cluster(test_labels.cpu().numpy(), im2cluster.cpu().numpy())
    return_dict = dict(nmi=nmi, ami=ami, acc=acc)
    return return_dict


if __name__ == '__main__':
    args = ClusterOptions().parse()
    init_distributed_mode(args)
    cudnn.benchmark = True
    val_results = run(args)
    
    val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_results.items()])
    if get_rank() == 0:
        # print("result on train: {}".format(train_str))
        print("result on val: {}".format(val_str))
    dist.barrier()
