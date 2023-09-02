import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps


def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()


def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)


def main(gpu, cfg, profile=False):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb) 
        writer = SummaryWriter(log_dir=cfg.run_dir) #指定日志的存储路径
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True #启用CuDNN库
    logging.info(cfg) #它用于将一条信息写入日志文件或打印到控制台

    if not cfg.model.get('criterion_args', False): #这是一个字典的操作，属性存在返回值，不存在返回False
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank) #得到模型
    model_size = cal_model_parm_nums(model) #计算模型大小
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels #设置in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer) #创建optimizer
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(
        val_loader.dataset, 'classes') else None or np.range(num_classes)
    validate_fn = eval(cfg.get('val_fn', 'validate')) #validate

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    # ===> start training
    """
    val_macc: 验证集上的平均分类正确率(mean class accuracy)。

    val_oa: 验证集上的总体分类正确率(overall accuracy)。

    val_accs: 一个列表，用于存储验证集上每个类别的分类正确率。

    best_val: 最佳验证性能(通常指最高的验证分类正确率)。

    macc_when_best: 在达到最佳验证性能时的平均类别正确率。

    best_epoch: 达到最佳验证性能的训练轮次(epochs)。
    
    """
    val_macc, val_oa, val_accs, best_val, macc_when_best, best_epoch = 0., 0., [], 0., 0., 0
    model.zero_grad()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)#在分布式训练中，为了确保每个训练轮次的数据分布都不同，需要设置每个数据加载器（train_loader）的采样器（sampler）的 epoch 属性。这样可以使每个计算节点上的数据加载器在每个轮次中加载不同的数据子集。
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        train_loss, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader,
                            optimizer, scheduler, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_macc, val_oa, val_accs, val_cm = validate_fn(
                model, val_loader, cfg)
            is_best = val_oa > best_val
            if is_best:
                best_val = val_oa
                macc_when_best = val_macc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
                print_cls_results(val_oa, val_macc, val_accs, epoch, cfg)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_oa {train_oa:.2f}, val_oa {val_oa:.2f}, best val oa {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_oa', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('mAcc_when_best', macc_when_best, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
    # test the last epoch
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, best_epoch)
        writer.add_scalar('test_macc', test_macc, best_epoch)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)

    if writer is not None:
        writer.close()
    dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True) #将每个值都移动到cuda上
        num_iter += 1
        points = data['x']
        target = data['y']
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
        num_curr_pts = points.shape[1] #shape = (32,2048,4) , (batch_size, num_points, num_features)
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(
                points[:, :, :3].contiguous(), point_all) #最远点采样point_all个点.(batch_size, point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]#从point_all个点中随机采样npoints个点，(batch_size, npoints)
            points = torch.gather(
                points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))#根据最远点采样得到的索引 fps_idx，从原始的点云数据 points 中选择和收集对应的点，形成一个新的点云数据。

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()#(32,4,1024),(batch_size, num_channels, num_points)
        logits, loss = model.get_logits_loss(data, target) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target)#logits代表推理的结果，loss代表损失函数的值
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2) #执行梯度裁剪
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item()) #更新loss_meter
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        points = data['x']
        points = points[:, :npoints]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    tp, count = cm.tp, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return macc, overallacc, accs, cm
