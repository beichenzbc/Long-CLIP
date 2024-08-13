import torch
#from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
#the original concat_all_gather is abandoned because of no gradient backward
from utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

import sys
sys.path.append("..")

from sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import longclip

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import os
import subprocess
import collections
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torch.cuda.amp import GradScaler
# import warnings
# warnings.filterwarnings("ignore")


class CLIP_Clean_Train():
    def __init__(self, rank,local_rank,args):
        self.rank=rank
        self.local_rank = local_rank
        self.base_model = args.base_model
        self.model, _ = longclip.load_from_clip(self.base_model, device='cpu',download_root=args.download_root)
        self.model.train()
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
        self.model = self.model.cuda()
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        if args.exp_name == "auto":
            self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
        else:
            self.logdir = args.exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
           
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler =GradScaler()

    # move PCA to CLIP class
    # def PCA(self, input_tensor, PCA_dim):
    #     mean = torch.mean(input_tensor, dim=0)
    #     X_centered = input_tensor - mean.unsqueeze(0)
    #     X_centered = X_centered.float()
    #     cov_matrix = torch.mm(X_centered.T, X_centered)
    #     eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
    #     eigenvalues = eigenvalues.float()
    #     eigenvectors = eigenvectors.float()    
    #     sorted_indices = torch.argsort(eigenvalues, descending=True)
    #     eigenvectors = eigenvectors[:, sorted_indices]
    #     principal_components = eigenvectors[:, :PCA_dim]
    #     X_transformed = torch.mm(X_centered, principal_components)
    #     X_reversed = torch.mm(X_transformed, principal_components.T)
    #     X_reversed += mean
    #     return X_reversed

    #rewrite forward in CLIP class to take place inference function, such that DDP will be effective
    # def inference(self, images, texts):
    #     image_features = self.model.module.encode_image(images)
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     text_features = self.model.module.encode_text(texts)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    #     image_feat_all = concat_all_gather(image_features)
    #     text_feat_all = concat_all_gather(text_features)
        
    #     sim_i2t = torch.matmul(image_features, text_feat_all.T)
    
    #     sim_t2i = torch.matmul(image_feat_all, text_features.T)
    #     sim_t2i = sim_t2i.T
        
    #     sim_i2t = self.model.logit_scale.exp() * sim_i2t
    #     sim_t2i = self.model.logit_scale.exp() * sim_t2i

    #     if is_dist_avail_and_initialized():
    #         rank = dist.get_rank()
    #     else:
    #         rank = 0
    #     bs = images.size(0)
    #     targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
    #         images.device
    #     )
        
    #     loss_itc = (
    #             F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
    #             + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
    #         ) / 2
    #     return loss_itc


    # def inference_short(self, images, texts):
    #     image_features = self.model.module.encode_image(images)
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     image_features = self.PCA(image_features, 32)

    #     text_features = self.model.module.encode_text(texts)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    #     image_feat_all = concat_all_gather(image_features)
    #     text_feat_all = concat_all_gather(text_features)

    #     sim_i2t = torch.matmul(image_features, text_feat_all.T)
    #     sim_t2i = torch.matmul(image_feat_all, text_features.T)
    #     sim_t2i = sim_t2i.T
        
    #     sim_i2t = self.model.logit_scale.exp() * sim_i2t
    #     sim_t2i = self.model.logit_scale.exp() * sim_t2i

    #     if is_dist_avail_and_initialized():
    #         rank = dist.get_rank()
    #     else:
    #         rank = 0
    #     bs = images.size(0)
    #     targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
    #         images.device
    #     )
        
    #     loss_itc = (
    #             F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
    #             + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
    #         ) / 2
    #     return loss_itc

    def train_epoch(self, dataloader, epoch, start_iter=0):
        running_loss = 0.0
        running_loss_short = 0.0
        #rank = torch.distributed.get_rank() 
        num_batches_per_epoch = len(dataloader)
        for i, (images, texts, short_text) in enumerate(tqdm(dataloader, disable=(self.rank != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            #images = images.cuda()
            #images_short = images.clone()
            texts = longclip.tokenize(texts, truncate=True).cuda()
            short_text = longclip.tokenize(short_text, truncate=True).cuda()
            self.scheduler(step)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_long,loss_short = self.model(images, texts,short_text,self.rank)
            # try:
            #     loss_short = 0.1 * self.inference_short(images_short, short_text)
            #     loss.backward()
            #     loss_short.backward()
                
            # except:
            #     # SVD may encounter infs, very rare occasion.
            #     loss.backward()
                loss=loss_long+loss_short
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        # ToDo: revise the report part
        #     running_loss += loss.item()
        #     running_loss_short += loss_short.item()
        #     batch_num = i
            
        #     loss = running_loss
        #     running_loss = 0.0

        #     loss_short = running_loss_short
        #     running_loss_short = 0.0

        #     loss = torch.tensor(loss).cuda()
        #     dist.all_reduce(loss)
        #     loss = loss.item() / torch.distributed.get_world_size()

        #     loss_short = torch.tensor(loss_short).cuda()
        #     dist.all_reduce(loss_short)
        #     loss_short = loss_short.item() / torch.distributed.get_world_size()

        #     rank = torch.distributed.get_rank()
        #     if step % 100 == 0:
        #         if rank == 0:
        #             self.writer.add_scalar("hyper/lr", self.optimizer.param_groups[0]['lr'], step)
        #             self.writer.add_scalar("logit_scale/train", self.model.logit_scale.item(), step)
        #             print("=====================================")
        #             print(f"train lr step {step}: {self.optimizer.param_groups[0]['lr']}")
        #             print(f"train logit_scale step {step}: {self.model.logit_scale.item()}")
        #             print(f"train loss step {step}: {loss}")
        #             print(f"train loss short step {step}: {loss_short}")
        #             print("=====================================")
        #             self.writer.add_scalar("Loss/train", loss + loss_short, step)
                    
        #             with torch.no_grad():
        #                 self.model.eval()
        #                 self.test(epoch = epoch)
        #                 self.model.train()

        # return running_loss / batch_num

    @torch.no_grad()
    def test_epoch(self, dataloader):
        temp_corr_dict = dict()
        rank = torch.distributed.get_rank()

        for id, (images, text) in enumerate(tqdm(dataloader, disable=(rank != 0))):

            images = images.cuda()
            image_features = self.model.module.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text = longclip.tokenize(text, truncate=True).cuda()
            text_feature = self.model.module.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

            i = 0
            correct = 0
            total = 0

            for i in range(text_feature.shape[0]):
                text = text_feature[i]
                sim = text @ image_features.T
                sim = sim.squeeze()
                correct_i = torch.argmax(sim)

                if i==correct_i:
                    correct = correct + 1
                total = total + 1

        return correct/total
    
    def test(self, epoch=0):
        rank = torch.distributed.get_rank()
        if rank == 0:
            self.model.eval()
            testset = share4v_val_dataset()
            testloader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=32, pin_memory=True)
            with torch.no_grad():    

                acc = self.test_epoch(testloader)
                print("=====================================")
                print(f"test mean of share4v retrieval: {acc}")
                print("=====================================")

            return
    
    def train(self, resume=False, warmup_length=200):
        trainset = share4v_train_dataset()
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=32, pin_memory=True)

        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader))
        start_epoch = 0
        resume_iter = 0
        
        for epoch in range(start_epoch, self.num_epoch):
            
            self.train_epoch(train_loader, epoch, start_iter=resume_iter)
            if self.rank == 0:
                name = "longclip.pt"
                now = datetime.now()
                formatted_date = now.strftime("%m-%d--%H_%M_%S_")
                #torch.distributed.barrier()
                torch.save(self.model.module.state_dict(), './checkpoints/'+str(self.rank)+formatted_date+name)
            # print("=====================================")
            # print(f"loss after training epoch: {epoch}")
            # print("=====================================")

            # if epoch == self.num_epoch - 1:
            #     if self.base_model == "ViT-B/16":
            #         name = 'longclip-B.pt'
            #     elif self.base_model == "ViT-L/14":
            #         name = 'longclip-L.pt'
            #     else:
            #         name = "longclip-others.pt"

            #     torch.save(self.model.module.state_dict(), name)

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
    return rank, rank % num_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-L/14", help="CLIP Base Model")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size per gpu."#112
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action='store_true',
        help="resume training from checkpoint."
    )
    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
    args = parser.parse_args()
    rank,local_rank = setup_distributed()
    print("DDP Done")

    trainer = CLIP_Clean_Train(
        rank=rank,
        local_rank=local_rank, 
        args=args
        )
    trainer.train(resume=args.resume, warmup_length=args.warmup_length)
