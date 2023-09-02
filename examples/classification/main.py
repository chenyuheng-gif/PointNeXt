import __init__
import os, argparse, yaml, numpy as np
from torch import multiprocessing as mp
from examples.classification.train import main as train
from examples.classification.pretrain import main as pretrain
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args() #解析命令行参数，已知的参数将被解析到args中，未知的参数将被解析到opts列表中
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True) #遍历式的加载yaml文件
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)#cfg.rank: 这是表示当前进程在分布式环境中的排名（或编号）。在分布式训练中，每个计算节点都会被分配一个唯一的排名，用来区分不同的节点。
    cfg.sync_bn = cfg.world_size > 1# cfg.mp: 这很可能是指"multiprocessing"（多进程）的缩写，用于表示代码是否使用了多进程并行。在分布式训练中，通常会使用多进程来并行地训练多个模型副本。
 #   cfg.sync_bn: 这是一个同步批归一化（Sync Batch Normalization）的标志位。在分布式训练中，由于每个节点处理不同的数据子集，使用批归一化时可能需要同步不同节点的统计信息，以保证模型的训练稳定性和收敛性。
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]#"scanobjectnn"
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]#"pointnet++"
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    if cfg.mode in ['resume', 'val', 'test']: #从预训练的程序中加载模型
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:  # resume from the existing ckpt and reuse the folder.
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags #给wandb增加一个tag
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2) #将配置对象cfg写进f中，使用缩进为2的格式
        os.system('cp %s %s' % (args.cfg, cfg.run_dir)) #将args.cfg复制到cfg.run_dir中
    cfg.cfg_path = cfg_path
    cfg.wandb.name = cfg.run_name

    if cfg.mode == 'pretrain':
        main = pretrain
    else:
        main = train

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg, args.profile))
    else:
        main(0, cfg, profile=args.profile)#进入训练函数 ,profile参数用于启用性能分析
