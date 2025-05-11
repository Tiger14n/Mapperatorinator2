import multiprocessing
import time
from multiprocessing.managers import Namespace

import torch
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import (
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)

from ..dataset import OrsDataset, OsuParser
from ..dataset.mmrs_dataset import MmrsDataset
from ..event import EventType
from ..model.configuration_mapperatorinator2 import MapperatorinatorConfig
from ..model.modeling_mapperatorinator2 import Mapperatorinator
from ..tokenizer import Tokenizer
from ..config import TrainConfig


def get_shared_training_state() -> Namespace:
    mgr = multiprocessing.Manager()
    shared = mgr.Namespace()
    shared.current_train_step = 1
    shared.current_epoch = 1
    shared.last_log = time.time()
    shared.current_loss = np.inf
    shared.best_loss = np.inf
    return shared


def get_model_config(args: TrainConfig, tokenizer: Tokenizer) -> MapperatorinatorConfig:
    return MapperatorinatorConfig(
        backbone_model_name=args.model.name,
        backbone_overwrite=args.model.overwrite,
        backbone_add_config=args.model.add_config,
        flash_attention=args.flash_attention,
        vocab_size_in=tokenizer.vocab_size_in,
        vocab_size_out=tokenizer.vocab_size_out,
        num_classes=tokenizer.num_classes,
        input_features=args.model.input_features,
        embed_decoder_input=args.model.embed_decoder_input,
        do_style_embed=args.model.do_style_embed,
        sample_rate=args.model.spectrogram.sample_rate,
        n_fft=args.model.spectrogram.n_fft,
        n_mels=args.model.spectrogram.n_mels,
        hop_length=args.model.spectrogram.hop_length,
        rhythm_weight=args.data.rhythm_weight,
        rhythm_token_start=tokenizer.event_start[EventType.TIME_SHIFT],
        rhythm_token_end=tokenizer.event_end[EventType.TIME_SHIFT],
        src_seq_len=args.data.src_seq_len,
        tgt_seq_len=args.data.tgt_seq_len,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.sos_id,
        eos_token_id=tokenizer.eos_id,
        decoder_start_token_id=tokenizer.sos_id,
        max_length=args.data.tgt_seq_len,
    )


def get_model(args: TrainConfig, tokenizer: Tokenizer) -> Mapperatorinator:
    model = Mapperatorinator(get_model_config(args, tokenizer))
    return model



def get_tokenizer(args: TrainConfig) -> Tokenizer:
    return Tokenizer(args)


def get_optimizer(model: Mapperatorinator, args: TrainConfig) -> Optimizer:
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == 'adamw':
        from torch.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adamwscale':
        from .copied_utils import AdamWScale
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adafactor':
        from torch.optim import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(optimizer: Optimizer, args: TrainConfig, accelerator) -> LRScheduler:
    scheduler_p1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps * accelerator.num_processes,
        last_epoch=-1,
    )

    scheduler_p2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps * accelerator.num_processes - args.optim.warmup_steps * accelerator.num_processes,
        eta_min=args.optim.final_cosine,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_p1, scheduler_p2],
        milestones=[args.optim.warmup_steps * accelerator.num_processes],
    )

    return scheduler


def get_dataset(args: TrainConfig, test: bool, **kwargs) -> Dataset:
    if args.data.dataset_type == "ors":
        return OrsDataset(args=args.data, test=test, **kwargs)
    elif args.data.dataset_type == "mmrs":
        return MmrsDataset(args=args.data, **kwargs)
    else:
        raise NotImplementedError


def get_dataloaders(tokenizer: Tokenizer, args: TrainConfig, shared: Namespace) -> tuple[DataLoader, DataLoader]:
    parser = OsuParser(args, tokenizer)
    dataset = {
        "train": get_dataset(
            args=args,
            test=False,
            parser=parser,
            tokenizer=tokenizer,
            shared=shared,
        ),
        "test": get_dataset(
            args=args,
            test=True,
            parser=parser,
            tokenizer=tokenizer,
            shared=shared,
        ),
    }

    dataloaders = {}
    for split in ["train", "test"]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            num_workers=args.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.dataloader.num_workers > 0,
            worker_init_fn=worker_init_fn,
        )

    return dataloaders["train"], dataloaders["test"]


def worker_init_fn(worker_id: int) -> None:
    """
    Give each dataloader a unique slice of the full dataset.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        np.ceil((overall_end - overall_start) / float(worker_info.num_workers)),
    )
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
