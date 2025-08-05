import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class PastAllPredTriMask:
    def __init__(self, B, past_len, pred_len, device="cpu"):
        # seq_len = past_len + pred_len
        # mask_shape = [B, 1, seq_len, seq_len]
        with torch.no_grad():
            # [B,1,pred_len,pred_len]
            mask_tri = torch.triu(torch.ones([B, 1, pred_len, pred_len], dtype=torch.bool), diagonal=1).to(device)
            # [B,1,pred_len,seq_len]
            mask_tri = torch.cat((torch.zeros((B, 1, pred_len, past_len), dtype=torch.bool).to(device), mask_tri),
                                 dim=3)
            # print(mask_tri.shape)

            # [B,1,past_len,past_len]
            mask_all = torch.zeros((B, 1, past_len, past_len), dtype=torch.bool).to(device)
            # [B,1,past_len,seq_len]
            mask_all = torch.cat((mask_all, torch.ones((B, 1, past_len, pred_len), dtype=torch.bool).to(device)), dim=3)
            # print(mask_all.shape)

            self._mask = torch.cat((mask_all, mask_tri), dim=2)

    @property
    def mask(self):
        return self._mask


class PastAllMask:
    def __init__(self, B, past_len, pred_len, device="cpu"):
        # seq_len = past_len + pred_len
        # mask_shape = [B, 1, seq_len, seq_len]
        with torch.no_grad():
            # [B,1,pred_len,pred_len]
            mask_pred = torch.zeros([B, 1, pred_len, past_len], dtype=torch.bool).to(device)
            # [B,1,pred_len,seq_len]
            mask_pred = torch.cat((mask_pred, torch.ones((B, 1, pred_len, pred_len), dtype=torch.bool).to(device)),
                                  dim=3)
            # print(f'来自PastAllMask的mask_pred shape is {mask_pred.shape}')
            # print(mask_pred)

            # [B,1,past_len,past_len]
            mask_all = torch.zeros((B, 1, past_len, past_len), dtype=torch.bool).to(device)
            # [B,1,past_len,seq_len]
            mask_all = torch.cat((mask_all, torch.ones((B, 1, past_len, pred_len), dtype=torch.bool).to(device)), dim=3)
            # print(mask_all.shape)

            self._mask = torch.cat((mask_all, mask_pred), dim=2)

    @property
    def mask(self):
        return self._mask