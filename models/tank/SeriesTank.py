import numpy as np
import torch
import torch.nn as nn

from utils.Embed import DataEmbedding
from utils.attention import MultiHeadAttention, FeedForward, sacffn
from utils.layers import EncoderLayer, Encoder, Encoder_origin
from utils.masking import PastAllPredTriMask

class SeriesTank(nn.Module):
    def __init__(self, cfg):
        super(SeriesTank, self).__init__()
        self.batch_size = cfg['batch_size']
        self.past_len = cfg['past_len']
        self.pred_len = cfg['pred_len']
        self.src_size = cfg['src_size']
        if cfg['use_runoff'] == True:
            self.src_size = self.src_size + 1
            print("runoff")
        self.tgt_size = cfg['tgt_size']
        self.static_size = cfg['static_size']
        self.n_heads = cfg['n_heads']
        self.d_model = cfg['d_model']  # d_model==hidden_size
        self.d_ff = cfg['d_ff']
        self.batch_size = cfg['batch_size']
        self.dropout = cfg['dropout']

        self.device = cfg['device']
        self.use_var = cfg['use_var']
        self.sub_model = cfg['sub_model']
        self.runoff = cfg['use_runoff']
        self.cfg = cfg

        # 目前采用的
        if cfg['sub_model'] == 'var_plus':
            print('var_plus')
            self.tank = ParameterPlusSeriesTank(self.past_len, self.pred_len, self.n_heads,
                                        self.d_model, self.d_ff, self.src_size + self.static_size, self.dropout)

    def forward(self, src, src_mark, train_x_mean=None, train_x_std=None, train_y_mean=None, train_y_std=None):
        # note: 缺失属性用0填充
        fea_mask_bool = torch.isnan(src)
        src[fea_mask_bool] = 0.0


        ep = src[:, :, 0:1]
        prcp = src[:, :, 1:2]
        others = src[:, :, 2:]


        if self.sub_model in ['var_plus','var_plus_s','var_plus_sp'] or self.sub_model == 'lstm':
            # STEP-1
            if train_x_mean is None and train_x_std is None:
                raise ValueError("need train x mean and std")

            out_tci, out_ep = self.tank(ep, prcp, others, src_mark, train_x_mean, train_x_std, 1)

            # STEP-2
            if train_y_mean is None and train_y_std is None:
                # NOTE:all
                raise ValueError("need train y mean and std")

            out_tci = (out_tci - train_y_mean[0]) / train_y_std[0]
            out_ep = (out_ep - train_y_mean[1]) / train_y_std[1]

            return out_tci, out_ep  # 结果输出先径流再蒸发

class ParameterPlusSeriesTank(nn.Module):
    def __init__(self,
                 # par, states,
                 # batch_size,
                 past_len, pred_len,
                 n_heads, d_model, d_ff,
                 input_size, dropout=0.0):
        super(ParameterPlusSeriesTank, self).__init__()

        # deep model setting
        self.d_model = d_model
        self.past_len = past_len
        self.pred_len = pred_len
        # self.batch_size = batch_size

        # input embedding
        self.known_embedding = DataEmbedding(input_size, d_model, embed_type='timeF', freq='d')
        # position embedding
        self.src_pos_emb = nn.Embedding(self.past_len + self.pred_len, self.d_model)
        self.src_linear = nn.Linear(input_size, self.d_model)
        self.max_storage = 1000
        self.tank_num = 4 # TODO choose tank num
        self.mul = 1
        self.OneTankParam = 12 + (self.tank_num - 2) * 4 + 2
        # 使用RRS率定参数
        self.AllParms_encoder = Encoder_origin(4, n_heads, d_model, d_ff, dropout_rate=dropout, mask_flag=False)
        self.AllParms_linear = nn.Linear(d_model, self.OneTankParam*self.mul)

    def _Reverse_Normalization(self, parms):
        b0 = (0.05, 1)
        b1 = (0.01, 100)
        b2 = (0.1, 0.5)
        b3 = (75, 100)
        b4 = (0.01, 50)
        b5 = (0.01, 0.5)
        b6 = (0, 100)
        b7 = (0, 5)
        b8 = (0, 0.5)
        b9 = (0,1)
        b11 = (0.01,10)
        tank_s = [b0,b2,b2,b2,b0,b0,b0,b0,b0,b0,b11,b11]
        tank_m = [b0,b5,b5,b0]
        tank_e = [b0,b5]

        # serial params
        for i in range(self.tank_num - 2):
            tank_s += tank_m
        tank_s += tank_e

        mean_weights = torch.tensor([(low + high) / 2 for (low, high) in tank_s]).to(parms.device)
        std_weights = torch.tensor([mean / 5 for mean in mean_weights]).to(parms.device)

        # rescale
        # dec_outputs为模型输出的参数
        dec_outputs = parms * std_weights + mean_weights

        return dec_outputs

    # Top Tank water balance
    def Soil_Moisture_Balance(self, tank_storage, del_rf_et, tank_top_parms):

        storage = torch.zeros_like(tank_storage)
        storage[:, :, 1:] = tank_storage[:, :, 1:]
        # the primary soil moisture storage
        Xp = tank_top_parms[:, :, 6:7]
        # the maximum capacity of primary soil moisture storage
        Cp = tank_top_parms[:, :, 7:8]
        # the secondary soil moisture storage
        Xs = tank_top_parms[:, :, 8:9]
        # the maximum capacity of secondary soil moisture storage
        Cs = tank_top_parms[:, :, 9:10]

        Xp = torch.minimum(Xp, Cp)
        Xs = torch.minimum(Xs, Cs)

        # linear parameters for  exchanging water flow
        # between primary soil moisture storage and secondary soil moisture storage
        K1 = tank_top_parms[:, :, 10:11]
        K2 = tank_top_parms[:, :, 11:12]
        # Top Tank Intial Free Water
        FW = tank_storage[:, :, 0:1]
        #
        Xa = FW + Xp + del_rf_et
        Xa = torch.maximum(Xa, torch.tensor(0).to(tank_storage.device))
        # water exchange between the primary and secondary soil moisture storages
        T2 = K2 * (Xp/Cp - Xs/Cs)
        T2 = torch.where(T2 > 0,
                         torch.minimum(T2, Xa),
                         -torch.minimum(torch.abs(T2), Xs))
        Xa = Xa - T2
        # free water in the lower tanks, there is water supply T  to the primary soil moisture storage
        T1 = K1 * (1 - Xp/Cp)
        T1 = torch.maximum(T1, torch.tensor(0).to(tank_storage.device))
        T1 = torch.minimum(T1, tank_storage[:, :, 1:2])
        Xa = Xa + T1

        # Final Free Water
        Xf = torch.maximum(Xa - Cp, torch.tensor(0).to(tank_storage.device))

        storage[:, :, 0:1] = Xf

        return storage


    def forward(self, ep, prcp, others, src_mark, train_x_mean, train_x_std,
                dt=1):  # dt默认是1, DT是以天为单位的每个时间间隔的长度,用于计算fland1中的dinc
        batch_size = ep.shape[0]
        known = torch.cat((prcp, others), dim=2)  # 主要是上层做了没意义的数据切分，因为之前的测试
        # 使用RRS进行参数率定
        # 使用
        # known_emb = self.known_embedding(known, src_mark)

        # RRS率定参数
        #########################################################################################
        src_position = torch.tensor(range(self.past_len + self.pred_len)).unsqueeze(0).repeat(batch_size, 1).to(
            ep.device)
        src_inputs = self.src_pos_emb(src_position) + self.src_linear(known)
        enc_self_attn_mask = None
        AllParms_enc_outputs = self.AllParms_encoder(src_inputs, enc_self_attn_mask)
        AllParms_enc_outputs = AllParms_enc_outputs[:, -self.pred_len:, :]
        AllParms_enc_outputs = self.AllParms_linear(AllParms_enc_outputs)
        #########################################################################################

        # 通过最大值最小值估算出均值和方差计算出来的
        params = self._Reverse_Normalization(AllParms_enc_outputs)

        # T-start
        # tank_top_parms = torch.zeros((batch_size, self.pred_len, 12)).to(prcp.device)
        tank_top_parms = params[:, :, :12]

        tank_top_parms[:, :, [0, 4, 5, 6, 7, 8, 9]] = tank_top_parms[:, :, [0, 4, 5, 6, 7, 8, 9]] \
                                                       * self.max_storage

        t1_is_b_params = torch.zeros((batch_size, self.pred_len, self.tank_num - 2)).to(prcp.device)
        t1_boc_b_params = torch.zeros((batch_size, self.pred_len, self.tank_num - 2)).to(prcp.device)
        t1_soc_b_params = torch.zeros((batch_size, self.pred_len, self.tank_num - 2)).to(prcp.device)
        t1_soh_b_params = torch.zeros((batch_size, self.pred_len, self.tank_num - 2)).to(prcp.device)

        for j in range(self.tank_num - 2):
            t1_is_b_params[:, :, j:j + 1] = params[:, :,12 + 4 * j:13 + 4 * j]
            t1_boc_b_params[:, :, j:j + 1] = params[:, :,13 + 4 * j:14 + 4 * j]
            t1_soc_b_params[:, :, j:j + 1] = params[:, :,14 + 4 * j:15 + 4 * j]
            t1_soh_b_params[:, :, j:j + 1] = params[:, :,15 + 4 * j:16 + 4 * j]

        t1_is_b_params = t1_is_b_params * self.max_storage
        t1_soh_b_params = t1_soh_b_params * self.max_storage

        tank_bottom_parms = params[:, :, -2:]
        tank_bottom_parms[:, :, 0] = tank_bottom_parms[:, :, 0] * self.max_storage


        prcp = prcp * train_x_std[1] + train_x_mean[1]
        prcp = prcp[:, -self.pred_len:,:]
        prcp = torch.clamp(prcp, min=0.0)
        ep = ep * train_x_std[0] + train_x_mean[0]
        ep = ep[:, -self.pred_len:,:]
        ep = torch.clamp(ep, min=0.0)

        # Difference of precipitation & evapotranspiration [only inflow to Tank 0]
        del_rf_et = torch.relu(prcp - ep)
        tank_storage = torch.zeros((batch_size, self.pred_len, self.tank_num)).to(prcp.device)
        side_outlet_flow = torch.zeros((batch_size, self.pred_len, self.tank_num)).to(prcp.device)
        bottom_outlet_flow = torch.zeros((batch_size, self.pred_len, self.tank_num-1)).to(prcp.device)

        # Top tank
        tank_storage[:, :, 0:1] = torch.maximum(tank_top_parms[:, :, 0:1],
                                                   torch.tensor(0.0).to(prcp.device))
        # Body tanks
        tank_storage[:, :, 1:self.tank_num - 1] = torch.maximum(t1_is_b_params,
                                                                   torch.tensor(0.0).to(prcp.device))
        # Bottom tank
        tank_storage[:, :, -1:] = torch.maximum(tank_bottom_parms[:, :, 0:1],
                                                   torch.tensor(0.0).to(prcp.device))


        tank_storage = self.Soil_Moisture_Balance(tank_storage, del_rf_et, tank_top_parms)

        # Top tank
        bottom_outlet_flow[:, :, 0:1] = tank_top_parms[:, :, 1:2] * tank_storage[:, :, 0:1]
        # Body tanks
        bottom_outlet_flow[:, :, 1:self.tank_num - 1] = t1_boc_b_params \
                                                           * tank_storage[:, :, 1:self.tank_num - 1]

        for i in range(self.tank_num - 1):
            left_storage = tank_storage[:, :, i] - bottom_outlet_flow[:, :, i]
            increment_storage = tank_storage[:, :, i + 1] + bottom_outlet_flow[:, :, i]

            new_tank_storage = tank_storage.clone()
            new_tank_storage[:, :, i] = left_storage
            new_tank_storage[:, :, i + 1] = increment_storage

            tank_storage = new_tank_storage

        # TANK 0 : surface runoff
        side_outlet_flow[:, :, 0:1] = tank_top_parms[:, :, 3:4] * torch.maximum(tank_storage[:, :, 0:1] - tank_top_parms[:, :, 5:6],
                                                       torch.tensor(0.0).to(prcp.device)) \
                                 + tank_top_parms[:, :, 2:3] * torch.maximum(tank_storage[:, :, 0:1] - tank_top_parms[:, :, 4:5],
                                                             torch.tensor(0.0).to(prcp.device))

        side_outlet_flow[:, :, 1:self.tank_num-1] = t1_soc_b_params * \
                                                    torch.maximum(tank_storage[:, :, 1:self.tank_num-1] - t1_soh_b_params,
                                                             torch.tensor(0.0).to(prcp.device))

        # TANK 3 : base-flow | Side outlet height = 0
        side_outlet_flow[:, :, -1:] = tank_bottom_parms[:, :, -1:] * tank_storage[:, :, -1:]

        discharge = torch.sum(side_outlet_flow, dim=2, keepdim=True)
        ep = torch.zeros_like(discharge)

        return discharge, ep
