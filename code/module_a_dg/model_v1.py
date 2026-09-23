import torch.nn as nn
import torch.utils.data
import numpy as np
import torch


def pearson_batch(x, y):
    numerator = ((x - x.mean(dim=1, keepdim=True)) *
                 (y - y.mean(dim=1, keepdim=True))).sum(dim=1) / (x.shape[1] - 1)
    denominator = x.std(dim=1) * y.std(dim=1)
    r = numerator / (denominator + 1e-12)
    return r


class BrainNetworkCL(nn.Module):
    def __init__(self, in_dim=204, att_out_dim=192, out_dim=768, temprature=0.05,
                 train_subs=None, use_weight=False, dropout=0.5,
                 negsample_num=32, cl_len=None, kernel_size=10, device='cpu',
                 embeds=None, feat_frames_id=None):
        super().__init__()

        self.BrainNetwork = BrainNetwork(in_dim=in_dim,
                                         att_out_dim=att_out_dim,
                                         out_dim=out_dim,
                                         kernel_size=kernel_size,
                                         dropout=dropout,
                                         device=device)
        self.negsample_num = negsample_num
        self.cl_len = cl_len
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.temprature = temprature
        if train_subs is None:
            self.trin_sub_ids = None
        else:
            self.trin_sub_ids = [i - 1 for i in train_subs]
        if use_weight:
            self.feature_weights = torch.nn.Parameter(torch.randn((out_dim), device=device), requires_grad=True)
        else:
            self.feature_weights = torch.ones((out_dim), device=device)
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels=out_dim, out_channels=2 * out_dim, kernel_size=1),
            nn.Conv1d(in_channels=2 * out_dim, out_channels=out_dim, kernel_size=1),
            nn.GELU(),
        )
        # Dynamically set time_projection target from actual stimulus time dimension.
        # envelope (100 Hz * 5 s) → 500; mel / bert / wav2vecbase9 (50 Hz * 5 s) → 250.
        stim_time_dim = embeds.shape[1] if embeds is not None else 500
        self.time_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(stim_time_dim)
        )

        # 不再依赖全局 mcfg.embeds / mcfg.feat_frames_id
        self.embeds = embeds
        self.feat_frames_id = feat_frames_id

    def _embeds_to_device(self):
        """将 embeds 移动到模型所在设备（惰性，仅在需要时调用）。"""
        if self.embeds is not None and self.embeds.device != self.BrainNetwork.device_used:
            self.embeds = self.embeds.to(self.BrainNetwork.device_used)

    def predict_label(self, x, sub_id, embed_set):
        sub_id = torch.ones_like(sub_id)
        pred = self.predict(x, sub_id)
        batchsize = x.shape[0]
        y_all = torch.zeros(embed_set.shape[0], embed_set.shape[1], embed_set.shape[3],
                            embed_set.shape[2]).to(self.BrainNetwork.device_used)
        for i in range(batchsize):
            embed_contrast = embed_set[:, i, :, :]
            embed_contrast = embed_contrast.transpose(1, 2).contiguous()
            y = self.projection(embed_contrast)
            y_all[:, i, :, :] = y
        pred = pred.transpose(1, 2).contiguous()
        pred = pred.repeat(embed_set.shape[0], 1, 1, 1)
        rr_all = pearson_batch(pred.view(-1, pred.shape[-1]), y_all.view(-1, y_all.shape[-1])).view(5, batchsize, -1)
        rr_all = rr_all * self.feature_weights.unsqueeze(0).unsqueeze(1).repeat(rr_all.shape[0], rr_all.shape[1], 1)
        rr_all_mean = torch.mean(rr_all, dim=2)
        pred_label = torch.argmax(rr_all_mean, dim=0)
        rr_all_mean = rr_all_mean.T
        return pred_label, rr_all_mean

    def predict(self, x, sub_id):
        pred = self.BrainNetwork.predict(x, sub_id)
        pred = self.time_projection(pred.transpose(1, 2)).transpose(1, 2).contiguous()
        return pred

    def forward(self, x, sub_id, frame_id, embed, sti_id):
        self._embeds_to_device()

        sub_id = torch.ones_like(sub_id)
        embed = embed.transpose(1, 2).contiguous()
        pred = self.predict(x, sub_id)
        pred = pred.transpose(1, 2).contiguous()
        batchsize = x.shape[0]

        sample_num = self.negsample_num
        device = self.BrainNetwork.device_used
        all_sim = torch.zeros(batchsize, embed.shape[1], sample_num + 1).to(device)

        for i in range(batchsize):
            frame_id_i = frame_id[i].item()
            sti_id_i = sti_id[i].item()

            frame_id_set = self.feat_frames_id[sti_id_i]
            frame_id_set = frame_id_set[frame_id_set != frame_id_i]

            if len(frame_id_set) > sample_num:
                neg_segment_id = np.random.choice(frame_id_set, sample_num, replace=False)
            else:
                neg_segment_id = np.random.choice(frame_id_set, sample_num, replace=True)

            neg_segment_id = np.random.permutation(neg_segment_id)
            embeds_contrast = self.embeds[neg_segment_id].to(device)
            embeds_contrast = embeds_contrast.transpose(1, 2).contiguous()
            x_i = pred[i]
            y_i = embed[i]
            x_i = x_i.repeat(sample_num + 1, 1, 1)
            x_i = x_i.view(-1, x_i.shape[-1])
            y_i = torch.concat([y_i.unsqueeze(0), embeds_contrast], dim=0)
            y_i = self.projection(y_i)
            y_i = y_i.view(-1, y_i.shape[-1])
            all_sim[i, :, :] = pearson_batch(x_i, y_i).view(sample_num + 1, -1).T

        all_sim = all_sim * self.feature_weights.unsqueeze(0).unsqueeze(2).repeat(all_sim.shape[0], 1, all_sim.shape[2])
        all_sim = torch.sum(all_sim, dim=1)
        logit = -self.logsoftmax(all_sim)
        loss = logit[:, 0].clone()
        logit_rank = logit

        _, indices = torch.sort(logit_rank, dim=-1, descending=False)
        rank = torch.nonzero(indices == 0)[:, 1] + 1
        return pred, loss, rank

    def test(self, x, sub_id, frame_id, embed, sti_id):
        self._embeds_to_device()

        embed = embed.transpose(1, 2).contiguous()
        embed = self.projection(embed)
        sample_num = 4
        batchsize = x.shape[0]
        device = self.BrainNetwork.device_used
        y_all = torch.zeros(sample_num + 1, embed.shape[0], embed.shape[1], embed.shape[2]).to(device)

        for i in range(batchsize):
            frame_id_i = frame_id[i].item()
            sti_id_i = sti_id[i].item()
            frame_id_set = self.feat_frames_id[sti_id_i]
            frame_id_set = frame_id_set[frame_id_set != frame_id_i]
            neg_segment_id = np.random.choice(frame_id_set, sample_num, replace=False)
            embeds_contrast = self.embeds[neg_segment_id].to(device)
            embeds_contrast = embeds_contrast.transpose(1, 2).contiguous()
            embeds_contrast = self.projection(embeds_contrast)
            y = embed[i]
            y_all[:, i, :, :] = torch.concat([y.unsqueeze(0), embeds_contrast], dim=0)
        pred = self.predict(x, torch.ones_like(sub_id))
        pred = pred.transpose(1, 2).contiguous()
        pred = pred.repeat(sample_num + 1, 1, 1, 1)
        rr_all = pearson_batch(pred.view(-1, pred.shape[-1]), y_all.view(-1, y_all.shape[-1])).view(sample_num + 1, batchsize, -1)
        rr_all = rr_all * self.feature_weights.unsqueeze(0).unsqueeze(1).repeat(rr_all.shape[0], rr_all.shape[1], 1)
        rr_all_mean = torch.mean(rr_all, dim=2)
        indices = torch.argmax(rr_all_mean, dim=0)
        return indices


class BrainNetwork(nn.Module):
    def __init__(self, in_dim=204,
                 att_out_dim=192,
                 out_dim=768,
                 dropout=0.5,
                 kernel_size=10,
                 device='cpu'):
        super().__init__()

        self.device_used = device
        self.spatialattention = nn.Linear(in_dim, att_out_dim)
        self.pre_conv = nn.Conv1d(in_channels=att_out_dim, out_channels=att_out_dim, kernel_size=1, bias=False)
        self.FeatureEncoder = nn.Sequential(
            FeatureEncoderBlock(1, channels=[att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size=kernel_size, dropout=dropout),
            FeatureEncoderBlock(2, channels=[att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size=kernel_size, dropout=dropout),
            FeatureEncoderBlock(3, channels=[att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size=kernel_size, dropout=dropout),
            FeatureEncoderBlock(4, channels=[att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size=kernel_size, dropout=dropout),
            FeatureEncoderBlock(5, channels=[att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size=kernel_size, dropout=dropout))

        self.finconv1 = nn.Conv1d(in_channels=att_out_dim, out_channels=2 * att_out_dim, kernel_size=1)
        self.finconv2 = nn.Conv1d(in_channels=2 * att_out_dim, out_channels=out_dim, kernel_size=1)
        self.GELU = nn.GELU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def predict(self, x, sub_num):
        x = self.spatialattention(x)
        x = x.transpose(1, 2)
        x = self.pre_conv(x)
        x = self.FeatureEncoder(x)
        x = self.finconv1(x)
        x = self.finconv2(x)
        x = self.GELU(x)
        pred = x.transpose(1, 2)
        return pred


class FeatureEncoderBlock(nn.Module):
    def __init__(self, k, channels, kernel_size=10, dropout=0.5):
        super().__init__()
        self.k = k
        if kernel_size == 3:
            dilation1 = int(pow(2, (2 * self.k) % 5))
            dilation2 = int(pow(2, (2 * self.k + 1) % 5))
        else:
            dilation1 = int(pow(2, (self.k) % 4))
            dilation2 = int(pow(2, (self.k + 1) % 4))
        self.conv1 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, padding='same',
                               dilation=dilation1)
        self.conv2 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_size, padding='same', dilation=dilation2)
        self.conv3 = nn.Conv1d(in_channels=channels[2], out_channels=channels[3], kernel_size=kernel_size, padding='same', dilation=2)
        self.batch_norm1 = nn.BatchNorm1d(num_features=channels[1], eps=1e-05, momentum=0.1)
        self.batch_norm2 = nn.BatchNorm1d(num_features=channels[2], eps=1e-05, momentum=0.1)
        self.GELU = nn.GELU()
        self.GLU = nn.GLU(dim=-2)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        if self.k == 1:
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv3(x)
            x = self.GLU(x)
            x = self.dropout(x)
            return x
        else:
            input = x
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv3(x)
            x = self.GLU(x)
            x = self.dropout(x)
            return input + x
