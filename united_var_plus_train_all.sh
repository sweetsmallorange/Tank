#tank模型第一次测试
SHELL_FOLDER=$(readlink -f "$0")

#使用的时候注意date切换跟着forcing_type来
python -u main.py \
  --sh_file "$SHELL_FOLDER" \
  --stage 'train' \
  --model_id 'attr9_448SeriesTank_runoff_Transformer' \
  --model 'SeriesTank' \
  --sub_model 'var_plus' \
  --seed 1234 \
  --camels_root '/data2/zmz1/Camels' \
  --forcing_type 'united' \
  --basins_list_path '' \
  --loss 'nse' \
  --warm_up true \
  --epochs 200 \
  --past_len 15 \
  --pred_len 1 \
  --src_size 5 \
  --static_size 9 \
  --use_static true \
  --loss_all false \
  --use_var false \
  --gpu 1 \
  --learning_rate 0.001 \
  --dropout 0.1 \
  --drop_last false \
  --batch_size 2048 \
  --use_runoff true \
  --local_run_dir 'runs_paper' \
  --freq 'd' \
  --dss_config_path 'data/448basins_list.txt' \
  --dss_config_ds 'CAMELS-US'
