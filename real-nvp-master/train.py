"""Train Real NVP on MNIST with mask-pair sweep"""

import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

from models import RealNVP, RealNVPLoss
from models.real_nvp.coupling_layer import SpatialMaskType, ChannelMaskType
from tqdm import tqdm


def parse_mask_pairs(mask_str):
    pairs = []
    for pair in mask_str.split(","):
        spatial, channel = pair.split(":")
        pairs.append((
            SpatialMaskType(spatial.lower()),
            ChannelMaskType(channel.lower())
        ))
    return pairs


def main(args):
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'

    mask_pairs_list = parse_mask_pairs(args.mask_pairs)

    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    loss_fn = RealNVPLoss()

    all_results = {}

    for pair in mask_pairs_list:
        print("\n" + "="*60)
        print(f"Running for mask pair: {pair}")
        print("="*60)

        net = RealNVP(
            num_scales=2,
            in_channels=1,
            mid_channels=64,
            num_blocks=8,
            mask_pairs=[pair]
        )

        net = net.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(net, args.gpu_ids)
            cudnn.benchmark = args.benchmark

        optimizer = optim.Adam(
            util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g'),
            lr=args.lr
        )

        epoch_losses = []

        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('samples', exist_ok=True)

        for epoch in range(20):
            train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm)
            test_loss = test(epoch, net, testloader, device, loss_fn, args.num_samples)

            epoch_losses.append(test_loss)

            # =========================
            # SAVE CHECKPOINT
            # =========================
            state = {
                'epoch': epoch,
                'model_state_dict': net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
            }

            torch.save(state, f'checkpoints/{str(pair)}_epoch_{epoch}.pth')

            # =========================
            #
            # =========================
            samples = sample(net, 10, device)

            grid = torchvision.utils.make_grid(
                samples,
                nrow=5,   # 2 rows of 5
                padding=2,
                pad_value=255
            )

            torchvision.utils.save_image(
                grid,
                f'samples/{str(pair)}_epoch_{epoch}.png'
            )

        all_results[str(pair)] = epoch_losses

    print("\nFINAL RESULTS (Loss vs Epoch):")
    for k, v in all_results.items():
        print(k, ":", v)


# =========================
# TRAIN
# =========================

def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)

    net.train()
    loss_meter = util.AverageMeter()

    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)

            optimizer.zero_grad()

            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)

            loss_meter.update(loss.item(), x.size(0))

            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(
                loss=loss_meter.avg,
                bpd=util.bits_per_dim(x, loss_meter.avg)
            )
            progress_bar.update(x.size(0))


# =========================
# SAMPLE
# =========================

def sample(net, batch_size, device):
    z = torch.randn((batch_size, 1, 32, 32), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)
    return x


# =========================
# TEST
# =========================

def test(epoch, net, testloader, device, loss_fn, num_samples):
    net.eval()
    loss_meter = util.AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)

                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)

                loss_meter.update(loss.item(), x.size(0))

                progress_bar.set_postfix(
                    loss=loss_meter.avg,
                    bpd=util.bits_per_dim(x, loss_meter.avg)
                )
                progress_bar.update(x.size(0))

    return loss_meter.avg


# =========================
# ENTRY
# =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP Mask Pair Sweep')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--gpu_ids', default='[0]', type=eval)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--max_grad_norm', type=float, default=100.)
    parser.add_argument('--num_samples', default=64, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--weight_decay', default=5e-5, type=float)

    parser.add_argument(
        '--mask_pairs',
        type=str,
        default="checkerboard:half,diagonal:alternate",
        help="Comma-separated spatial:channel pairs"
    )

    main(parser.parse_args())