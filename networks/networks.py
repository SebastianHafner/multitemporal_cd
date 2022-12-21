import torch
import torch.nn as nn
from torch.nn.modules.padding import ReplicationPad2d
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from pathlib import Path
from utils import experiment_manager
from networks import network_parts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_network(cfg):
    if cfg.MODEL.TYPE == 'unet':
        model = UNet(cfg)
    elif cfg.MODEL.TYPE == 'siamdiff':
        model = SiamDiff(cfg)
    elif cfg.MODEL.TYPE == 'unetlstm':
        model = UNetLSTM(cfg)
    elif cfg.MODEL.TYPE == 'lunet':
        model = LUNet(cfg)
    elif cfg.MODEL.TYPE == 'alunet':
        model = ALUNet(cfg)
    elif cfg.MODEL.TYPE == 'convlstmnet':
        model = ConvLSTMNet(cfg)
    elif cfg.MODEL.TYPE == 'mtunetlstm':
        model = SmallMultiTaskUNetLSTM(cfg)
    else:
        raise Exception(f'Unknown network ({cfg.MODEL.TYPE}).')
    return nn.DataParallel(model)


def save_checkpoint(network, optimizer, epoch, step, cfg: experiment_manager.CfgNode, early_stopping: bool = False):
    if early_stopping:
        save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_early_stopping.pt'
    else:
        save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch: float, cfg: experiment_manager.CfgNode, device: torch.device, net_file: Path = None,
                    best_val: bool = False):
    net = create_network(cfg)
    net.to(device)

    if net_file is None:
        net_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    if best_val:
        net_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_early_stopping.pt'

    checkpoint = torch.load(net_file, map_location=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']


class UNet(nn.Module):
    def __init__(self, cfg):

        self._cfg = cfg

        n_channels = len(cfg.DATALOADER.SAR_BANDS) if cfg.DATALOADER.MODALITIES[0] == 'sar' else 3
        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = [64, 128, ]

        super(UNet, self).__init__()

        first_chan = topology[0]
        self.inc = network_parts.InConv(n_channels, first_chan, network_parts.ConvBlock)
        self.outc = network_parts.OutConv(first_chan, n_classes)

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan]  # topography upwards
        up_dict = OrderedDict()

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer

            layer = network_parts.Down(in_dim, out_dim, network_parts.ConvBlock)

            print(f'down{idx + 1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx + 1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]

            layer = network_parts.Up(in_dim, out_dim, network_parts.ConvBlock)

            print(f'up{idx + 1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, x: torch.tensor):
        x = torch.cat((x[:, 0, ], x[:, -1, ]), 1)

        x1 = self.inc(x)

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        # Upward U:
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        out = self.outc(x1)

        return out


class LUNet(nn.Module):
    def __init__(self, cfg):
        super(LUNet, self).__init__()

        self._cfg = cfg
        n_channels = len(cfg.DATALOADER.SAR_BANDS) if cfg.DATALOADER.MODALITIES[0] == 'sar' else 3
        n_classes = cfg.MODEL.OUT_CHANNELS
        patch_size = cfg.MODEL.PATCH_SIZE

        convlstm_kwargs = {
            'kernel_size': (3, 3),
            'padding': (1, 1),
        }

        self.convlstm1 = network_parts.ConvLSTM(n_channels, 16, (patch_size, patch_size), **convlstm_kwargs)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.convlstm2 = network_parts.ConvLSTM(16, 32, (patch_size // 2, patch_size // 2), **convlstm_kwargs)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.convlstm3 = network_parts.ConvLSTM(32, 64, (patch_size // 4, patch_size // 4), **convlstm_kwargs)
        self.up1 = nn.ConvTranspose3d(64, 64, (1, 2, 2), stride=(1, 2, 2))
        # self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.convlstm4 = network_parts.ConvLSTM(96, 32, (patch_size // 2, patch_size // 2), **convlstm_kwargs)
        self.up2 = nn.ConvTranspose3d(32, 32, (1, 2, 2), stride=(1, 2, 2))
        # self.up2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.convlstm5 = network_parts.ConvLSTM(48, 16, (patch_size, patch_size), ** convlstm_kwargs)
        self.conv6 = nn.Conv2d(16, n_classes, (3, 3), padding=(1, 1))

    def forward(self, x: torch.tensor):
        # (B, TS, C, H, W) -> (B, C, TS, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # encoder
        x1 = self.convlstm1(x)

        # down 1
        x2 = self.pool1(x1)
        x3 = self.convlstm2(x2)

        # down 2
        x4 = self.pool2(x3)
        x5 = self.convlstm3(x4)

        # up 1
        # x6 = []
        # for i in range(x.size(2)):
        #     x6.append(self.up1(x5[:, :, i]))
        # x6 = torch.stack(x6, dim=2)
        x6 = self.up1(x5)
        x7 = self.convlstm4(torch.cat((x3, x6), dim=1))

        # up 2
        # x8 = []
        # for i in range(x.size(2)):
        #     x8.append(self.up2(x7[:, :, i]))
        # x8 = torch.stack(x8, dim=2)
        x8 = self.up2(x7)
        x9 = self.convlstm5(torch.cat((x1, x8), dim=1))

        out = self.conv6(x9[:, :, -1])
        return out


class ALUNet(nn.Module):
    def __init__(self, cfg):
        super(ALUNet, self).__init__()

        self._cfg = cfg
        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        self.convlstm = network_parts.ConvLSTM(n_channels, 16, (3, 3), 1)
        self.outconv = nn.Conv2d(16, n_classes, 1)

    def forward(self, x: torch.tensor):
        x1, _ = self.convlstm(x)
        x1 = x1[0]
        out = self.outconv(x1)
        return out


class ConvLSTMNet(nn.Module):

    def __init__(self, cfg):
        super(ConvLSTMNet, self).__init__()

        self._cfg = cfg
        num_channels = len(cfg.DATALOADER.SAR_BANDS) if self.modalities[0] == 'sar' else 3
        n_classes = cfg.MODEL.OUT_CHANNELS
        patch_size = (cfg.MODEL.PATCH_SIZE, cfg.MODEL.PATCH_SIZE)
        num_kernels = 64
        num_layers = 3

        self.sequential = nn.Sequential()

        convlstm_kwargs = {
            'out_channels': num_kernels,
            'kernel_size': (3, 3),
            'padding': (1, 1),
            'frame_size': patch_size,
        }

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", network_parts.ConvLSTM(num_channels, **convlstm_kwargs)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for l in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{l}", network_parts.ConvLSTM(num_kernels, **convlstm_kwargs)
            )

            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
            )

            # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(num_kernels, n_classes, (3, 3), padding=(1, 1))

    def forward(self, X):
        # (B, TS, C, H, W) -> (B, C, TS, H, W)
        X = X.permute(0, 2, 1, 3, 4)
        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:, :, -1])

        return output


# U-net with LSTM units from https://ieeexplore.ieee.org/abstract/document/8900330
class UNetLSTM(nn.Module):
    def __init__(self, cfg):
        super(UNetLSTM, self).__init__()

        self.cfg = cfg
        self.patch_size = cfg.MODEL.PATCH_SIZE
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = len(cfg.DATALOADER.SAR_BANDS) if cfg.DATALOADER.MODALITIES[0] == 'sar' else 3
        self.Conv1 = network_parts.ConvBlock(ch_in=in_channels, ch_out=16)
        self.set1 = network_parts.ExtensionLSTM(16, self.patch_size, self.patch_size)

        self.Conv2 = network_parts.ConvBlock(ch_in=16, ch_out=32)
        self.set2 = network_parts.ExtensionLSTM(32, self.patch_size / 2, self.patch_size / 2)

        self.Conv3 = network_parts.ConvBlock(ch_in=32, ch_out=64)
        self.set3 = network_parts.ExtensionLSTM(64, self.patch_size / 4, self.patch_size / 4)

        self.Conv4 = network_parts.ConvBlock(ch_in=64, ch_out=128)
        self.set4 = network_parts.ExtensionLSTM(128, self.patch_size / 8, self.patch_size / 8)

        self.Conv5 = network_parts.ConvBlock(ch_in=128, ch_out=256)
        self.set5 = network_parts.ExtensionLSTM(256, self.patch_size / 16, self.patch_size / 16)

        self.Up5 = network_parts.UpConv(ch_in=256, ch_out=128)
        self.Up_conv5 = network_parts.ConvBlock(ch_in=256, ch_out=128)

        self.Up4 = network_parts.UpConv(ch_in=128, ch_out=64)
        self.Up_conv4 = network_parts.ConvBlock(ch_in=128, ch_out=64)

        self.Up3 = network_parts.UpConv(ch_in=64, ch_out=32)
        self.Up_conv3 = network_parts.ConvBlock(ch_in=64, ch_out=32)

        self.Up2 = network_parts.UpConv(ch_in=32, ch_out=16)
        self.Up_conv2 = network_parts.ConvBlock(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16, cfg.MODEL.OUT_CHANNELS, kernel_size=1, stride=1, padding=0)

    def encoder(self, x: torch.tensor):
        x1, xout = self.set1(self.Conv1, x, device)
        x2, xout = self.set2(nn.Sequential(self.Maxpool, self.Conv2), xout, device)
        x3, xout = self.set3(nn.Sequential(self.Maxpool, self.Conv3), xout, device)
        x4, xout = self.set4(nn.Sequential(self.Maxpool, self.Conv4), xout, device)
        x5, xout = self.set5(nn.Sequential(self.Maxpool, self.Conv5), xout, device)
        return x1, x2, x3, x4, x5

    def forward(self, x: torch.tensor):
        # (T, B, C, H, W)
        x = x.permute(1, 0, 2, 3, 4)

        # encoding path
        x1, x2, x3, x4, x5 = self.encoder(x)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((d5, x4), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# https://rcdaudt.github.io/
class SiamDiff(nn.Module):

    def __init__(self, cfg):
        super(SiamDiff, self).__init__()

        self.cfg = cfg

        in_channels = len(cfg.DATALOADER.SAR_BANDS) if cfg.DATALOADER.MODALITIES[0] == 'sar' else 3
        out_channels = cfg.MODEL.OUT_CHANNELS

        self.conv11 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, out_channels, kernel_size=3, padding=1)

        # self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor):
        """Forward method."""
        x_t1, x_t2 = x[:, 0, ], x[:, -1, ]

        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x_t1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x_t2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)

        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        # self.sm(x11d)
        return x11d


class MultiTaskUNetLSTM(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(MultiTaskUNetLSTM, self).__init__()

        self.cfg = cfg
        self.patch_size = cfg.MODEL.PATCH_SIZE
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = len(cfg.DATALOADER.SAR_BANDS) if cfg.DATALOADER.MODALITIES[0] == 'sar' else 3
        self.Conv1 = network_parts.ConvBlock(ch_in=in_channels, ch_out=16)
        self.set1 = network_parts.ExtensionLSTM(16, self.patch_size, self.patch_size)

        self.Conv2 = network_parts.ConvBlock(ch_in=16, ch_out=32)
        self.set2 = network_parts.ExtensionLSTM(32, self.patch_size / 2, self.patch_size / 2)

        self.Conv3 = network_parts.ConvBlock(ch_in=32, ch_out=64)
        self.set3 = network_parts.ExtensionLSTM(64, self.patch_size / 4, self.patch_size / 4)

        self.Conv4 = network_parts.ConvBlock(ch_in=64, ch_out=128)
        self.set4 = network_parts.ExtensionLSTM(128, self.patch_size / 8, self.patch_size / 8)

        self.Conv5 = network_parts.ConvBlock(ch_in=128, ch_out=256)
        self.set5 = network_parts.ExtensionLSTM(256, self.patch_size / 16, self.patch_size / 16)

        self.Up5 = network_parts.UpConv(ch_in=256, ch_out=128)
        self.Up_conv5 = network_parts.ConvBlock(ch_in=256, ch_out=128)
        self.Up5_segm = network_parts.UpConv(ch_in=256, ch_out=128)
        self.Up_conv5_segm = network_parts.ConvBlock(ch_in=256, ch_out=128)

        self.Up4 = network_parts.UpConv(ch_in=128, ch_out=64)
        self.Up_conv4 = network_parts.ConvBlock(ch_in=128, ch_out=64)
        self.Up4_segm = network_parts.UpConv(ch_in=128, ch_out=64)
        self.Up_conv4_segm = network_parts.ConvBlock(ch_in=128, ch_out=64)

        self.Up3 = network_parts.UpConv(ch_in=64, ch_out=32)
        self.Up_conv3 = network_parts.ConvBlock(ch_in=64, ch_out=32)
        self.Up3_segm = network_parts.UpConv(ch_in=64, ch_out=32)
        self.Up_conv3_segm = network_parts.ConvBlock(ch_in=64, ch_out=32)

        self.Up2 = network_parts.UpConv(ch_in=32, ch_out=16)
        self.Up_conv2 = network_parts.ConvBlock(ch_in=32, ch_out=16)
        self.Up2_segm = network_parts.UpConv(ch_in=32, ch_out=16)
        self.Up_conv2_segm = network_parts.ConvBlock(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16, cfg.MODEL.OUT_CHANNELS, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_segm = nn.Conv2d(16, cfg.MODEL.OUT_CHANNELS, kernel_size=1, stride=1, padding=0)

    def encoder(self, x: torch.tensor):
        x1, xout = self.set1(self.Conv1, x, device)
        s1 = xout[0]
        a1 = xout[-1]

        x2, xout = self.set2(nn.Sequential(self.Maxpool, self.Conv2), xout, device)
        s2 = xout[0]
        a2 = xout[-1]

        x3, xout = self.set3(nn.Sequential(self.Maxpool, self.Conv3), xout, device)
        s3 = xout[0]
        a3 = xout[-1]

        x4, xout = self.set4(nn.Sequential(self.Maxpool, self.Conv4), xout, device)
        s4 = xout[0]
        a4 = xout[-1]

        x5, xout = self.set5(nn.Sequential(self.Maxpool, self.Conv5), xout, device)
        s5 = xout[0]
        a5 = xout[-1]

        return x1, x2, x3, x4, x5, s1, s2, s3, s4, s5, a1, a2, a3, a4, a5

    def decoder_lstm(self, x1, x2, x3, x4, x5):
        d5 = self.Up5(x5)
        d5 = torch.cat((d5, x4), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    def decoder_segm(self, s1, s2, s3, s4, s5):
        d5 = self.Up5_segm(s5)
        d5 = torch.cat((d5, s4), dim=1)
        d5 = self.Up_conv5_segm(d5)

        d4 = self.Up4_segm(d5)
        d4 = torch.cat((d4, s3), dim=1)
        d4 = self.Up_conv4_segm(d4)

        d3 = self.Up3_segm(d4)
        d3 = torch.cat((d3, s2), dim=1)
        d3 = self.Up_conv3_segm(d3)

        d2 = self.Up2_segm(d3)
        d2 = torch.cat((d2, s1), dim=1)
        d2 = self.Up_conv2_segm(d2)

        d1 = self.Conv_1x1_segm(d2)

        return d1

    def forward(self, x: torch.tensor):
        # (T, B, C, H, W)
        x = x.permute(1, 0, 2, 3, 4)

        x1, x2, x3, x4, x5, s1, s2, s3, s4, s5, a1, a2, a3, a4, a5 = self.encoder(x)

        d1 = self.decoder_lstm(x1, x2, x3, x4, x5)
        segm1 = self.decoder_segm(s1, s2, s3, s4, s5)
        segm2 = self.decoder_segm(a1, a2, a3, a4, a5)
        return d1, segm1, segm2


class SmallMultiTaskUNetLSTM(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(SmallMultiTaskUNetLSTM, self).__init__()

        self.cfg = cfg
        self.patch_size = cfg.MODEL.PATCH_SIZE
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = len(cfg.DATALOADER.SAR_BANDS) if cfg.DATALOADER.MODALITIES[0] == 'sar' else 3
        self.Conv1 = network_parts.ConvBlock(ch_in=in_channels, ch_out=16)
        self.set1 = network_parts.ExtensionLSTM(16, self.patch_size, self.patch_size)

        self.Conv2 = network_parts.ConvBlock(ch_in=16, ch_out=32)
        self.set2 = network_parts.ExtensionLSTM(32, self.patch_size / 2, self.patch_size / 2)

        self.Conv3 = network_parts.ConvBlock(ch_in=32, ch_out=64)
        self.set3 = network_parts.ExtensionLSTM(64, self.patch_size / 4, self.patch_size / 4)

        self.Up3 = network_parts.UpConv(ch_in=64, ch_out=32)
        self.Up_conv3 = network_parts.ConvBlock(ch_in=64, ch_out=32)
        self.Up3_segm = network_parts.UpConv(ch_in=64, ch_out=32)
        self.Up_conv3_segm = network_parts.ConvBlock(ch_in=64, ch_out=32)

        self.Up2 = network_parts.UpConv(ch_in=32, ch_out=16)
        self.Up_conv2 = network_parts.ConvBlock(ch_in=32, ch_out=16)
        self.Up2_segm = network_parts.UpConv(ch_in=32, ch_out=16)
        self.Up_conv2_segm = network_parts.ConvBlock(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16, cfg.MODEL.OUT_CHANNELS, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_segm = nn.Conv2d(16, cfg.MODEL.OUT_CHANNELS, kernel_size=1, stride=1, padding=0)

    def encoder(self, x: torch.tensor):
        x1, xout = self.set1(self.Conv1, x, device)
        s1 = xout[0]
        a1 = xout[-1]

        x2, xout = self.set2(nn.Sequential(self.Maxpool, self.Conv2), xout)
        s2 = xout[0]
        a2 = xout[-1]

        x3, xout = self.set3(nn.Sequential(self.Maxpool, self.Conv3), xout)
        s3 = xout[0]
        a3 = xout[-1]

        return x1, x2, x3, s1, s2, s3, a1, a2, a3,

    def decoder_lstm(self, x1, x2, x3):
        d3 = self.Up3(x3)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    def decoder_segm(self, s1, s2, s3):
        d3 = self.Up3_segm(s3)
        d3 = torch.cat((d3, s2), dim=1)
        d3 = self.Up_conv3_segm(d3)

        d2 = self.Up2_segm(d3)
        d2 = torch.cat((d2, s1), dim=1)
        d2 = self.Up_conv2_segm(d2)

        d1 = self.Conv_1x1_segm(d2)

        return d1

    def forward(self, x: torch.tensor):
        # (T, B, C, H, W)
        x = x.permute(1, 0, 2, 3, 4)

        x1, x2, x3, s1, s2, s3, a1, a2, a3 = self.encoder(x)

        d1 = self.decoder_lstm(x1, x2, x3)
        segm1 = self.decoder_segm(s1, s2, s3)
        segm2 = self.decoder_segm(a1, a2, a3)
        return d1, segm1, segm2
