# ========== AUS + BR + GB ==========
python src_configs_repo/change_src_configs.py -d noUS
nohup python -u src/train/Pretrain.py >noUS.txt&
nohup python -u src/test/PretrainTest.py >test_noUS.txt&

python src_configs_repo/change_src_configs.py -d noUS_DNN
nohup python -u src/train/Pretrain.py >noUS_DNN.txt&
nohup python -u src/test/PretrainTest.py >test_noUS_DNN.txt&

python src_configs_repo/change_src_configs.py -d noUS_LSTMS2S
nohup python -u src/train/Pretrain.py >noUS_LSTMS2S.txt&
nohup python -u src/test/PretrainTest.py >test_noUS_LSTMS2S.txt&

python src_configs_repo/change_src_configs.py -d noUS_past7
nohup python -u src/train/Pretrain.py >noUS_past7.txt&
nohup python -u src/test/PretrainTest.py >test_noUS_past7.txt&

python src_configs_repo/change_src_configs.py -d noUS_past30
nohup python -u src/train/Pretrain.py >noUS_past30.txt&
nohup python -u src/test/PretrainTest.py >test_noUS_past30.txt&

# ========== US ==========
# TRM: non-pretrained
python src_configs_repo/change_src_configs.py -d US
nohup python -u src/train/Pretrain.py >US.txt&
nohup python -u src/test/PretrainTest.py >test_US.txt&

python src_configs_repo/change_src_configs.py -d US_noStatic
nohup python -u src/train/Pretrain.py >US_noStatic.txt&
nohup python -u src/test/PretrainTest.py >test_US_noStatic.txt&

python src_configs_repo/change_src_configs.py -d US_2-7
nohup python -u src/train/Pretrain.py >US_2-7.txt&
nohup python -u src/test/PretrainTest.py >test_US_2-7.txt&

python src_configs_repo/change_src_configs.py -d US_1-7
nohup python -u src/train/Pretrain.py >US_1-7.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-7.txt&

python src_configs_repo/change_src_configs.py -d US_1-14
nohup python -u src/train/Pretrain.py >US_1-14.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-14.txt&

python src_configs_repo/change_src_configs.py -d US_1-28
nohup python -u src/train/Pretrain.py >US_1-28.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-28.txt&

python src_configs_repo/change_src_configs.py -d US_1-56
nohup python -u src/train/Pretrain.py >US_1-56.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-56.txt&

# TRM: pretrained + finetune (all and part)
python src_configs_repo/change_src_configs.py -d US_pretrained
nohup python -u src/train/Pretrain.py >US_pretrained.txt&
nohup python -u src/test/PretrainTest.py >test_US_pretrained.txt&

python src_configs_repo/change_src_configs.py -d US_2-7_pretrained
nohup python -u src/train/Pretrain.py >US_2-7_pretrained.txt&
nohup python -u src/test/PretrainTest.py >test_US_2-7_pretrained.txt&

python src_configs_repo/change_src_configs.py -d US_1-7_pretrained
nohup python -u src/train/Pretrain.py >US_1-7_pretrained.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-7_pretrained.txt&

python src_configs_repo/change_src_configs.py -d US_1-14_pretrained
nohup python -u src/train/Pretrain.py >US_1-14_pretrained.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-14_pretrained.txt&

python src_configs_repo/change_src_configs.py -d US_1-28_pretrained
nohup python -u src/train/Pretrain.py >US_1-28_pretrained.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-28_pretrained.txt&

python src_configs_repo/change_src_configs.py -d US_1-56_pretrained
nohup python -u src/train/Pretrain.py >US_1-56_pretrained.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-56_pretrained.txt&

# TRM: pretrained + non-finetune
python src_configs_repo/change_src_configs.py -d US_0-0_pretrained
nohup python -u src/train/Pretrain.py >US_0-0_pretrained.txt&
nohup python -u src/test/PretrainTest.py >test_US_0-0_pretrained.txt&


# LSTM: non-pretrained
python src_configs_repo/change_src_configs.py -d US_LSTMS2S
nohup python -u src/train/Pretrain.py >US_LSTMS2S.txt&
nohup python -u src/test/PretrainTest.py >test_US_LSTMS2S.txt&

python src_configs_repo/change_src_configs.py -d US_2-7_LSTMS2S
nohup python -u src/train/Pretrain.py >US_2-7_LSTMS2S.txt&
nohup python -u src/test/PretrainTest.py >test_US_2-7_LSTMS2S.txt&

python src_configs_repo/change_src_configs.py -d US_1-7_LSTMS2S
nohup python -u src/train/Pretrain.py >US_1-7_LSTMS2S.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-7_LSTMS2S.txt&

python src_configs_repo/change_src_configs.py -d US_1-14_LSTMS2S
nohup python -u src/train/Pretrain.py >US_1-14_LSTMS2S.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-14_LSTMS2S.txt&

python src_configs_repo/change_src_configs.py -d US_1-28_LSTMS2S
nohup python -u src/train/Pretrain.py >US_1-28_LSTMS2S.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-28_LSTMS2S.txt&

python src_configs_repo/change_src_configs.py -d US_1-56_LSTMS2S
nohup python -u src/train/Pretrain.py >US_1-56_LSTMS2S.txt&
nohup python -u src/test/PretrainTest.py >test_US_1-56_LSTMS2S.txt&
