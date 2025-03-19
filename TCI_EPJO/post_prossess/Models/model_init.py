from post_prossess.Models.NAFnet.naf_network import Network
from post_prossess.Models.loss_basic import mse_loss, l1_loss, LpipsLoss, WeightLoss
from post_prossess.Models.basic_model import BasicModel
from post_prossess.Models.SwinIR.network_swinir import SwinIR
from torch.utils.tensorboard import SummaryWriter


def def_model(net_name, device='cuda:0', save_root=None, demo_root=None, clear_root=None, blur_root=None,
              clear_val_root=None, blur_val_root=None, model_path=None, blur_val_root_no_gt=None):

    ema_scheduler = {
        "ema_start": 1,
        "ema_iter": 1,
        "ema_decay": 0.99
    }

    train = {
        "n_iter": 100000,
        "val_iter": 10000,
        "test_iter": 5000,
        "b_size": 4,
        "pre_b_size": 1,
        "img_size": 256
    }

    img_path = {
        "save_root": save_root,
        "demo_root": demo_root,
        "clear_root": clear_root,
        "blur_root": blur_root,
        "clear_val_root": clear_val_root,
        "blur_val_root": blur_val_root,
        "blur_val_root_no_gt": blur_val_root_no_gt
    }

    optimizer_datas = {"lr": 1e-4}
    model_net = None
    if net_name == 'NAFNet':
        net_config = {
            "width": 32,
            "enc_blk_nums": [1, 1, 1, 32],
            "middle_blk_num": 1,
            "dec_blk_nums": [1, 1, 1, 1]
        }
        model_net = Network(net_config).to(device)
    if net_name == 'SwinIR':
        upscale = 4
        window_size = 8
        height = 256 // upscale
        width = 256 // upscale
        model_net = SwinIR(upscale=upscale, img_size=(height, width), in_chans=48,
                           window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6],
                           embed_dim=96, num_heads=[6, 6, 6, 6, 6], mlp_ratio=4, upsampler='pixelshuffle').to(device)
        # model_net = SwinIR(upscale=upscale, img_size=(height, width), in_chans=48,
        #                    window_size=window_size, img_range=1., depths=[6, 6],
        #                    embed_dim=48, num_heads=[6, 6], mlp_ratio=4, upsampler='pixelshuffle').to(device)

    model = BasicModel(model_net, optimizer_datas, train, img_path, ema_scheduler=ema_scheduler,
                       device=device, model_path=model_path)
    model.writer = SummaryWriter(log_dir=demo_root + 'loss_dir')
    model.naf_iter = 0
    return model

