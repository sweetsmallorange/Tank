#tank模型第一次测试
SHELL_FOLDER=$(readlink -f "$0")
#使用的时候注意date切换跟着forcing_type来
python -u main.py \
  --sh_file "$SHELL_FOLDER" \
  --stage 'test' \
  --test_mode 'single' \
  --test_name '448basins' \
  --run_dir "/data2/zmz1/Tank/runs_paper/attr9_448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[90,1][2025062422]" \
  --model_id 'test' \
  --model 'SeriesTank' \
  --sub_model 'var_plus' \
  --seed 1234 \
  --camels_root '/data2/zmz1/Camels' \
  --forcing_type 'separated' \
  --basins_list_path '' \
  --loss 'mse' \
  --warm_up true \
  --epochs 200 \
  --past_len 90 \
  --pred_len 1 \
  --src_size 5 \
  --static_size 9 \
  --use_static true \
  --loss_all false \
  --use_var false \
  --gpu 0 \
  --learning_rate 0.001 \
  --dropout 0.1 \
  --drop_last false \
  --batch_size 2048 \
  --use_runoff True \
  --local_run_dir 'runs_paper' \
  --freq 'd' \
  --dss_config_path '/data2/zmz1/Tank/data/448basins_list.txt' \
  --dss_config_ds 'CAMELS-US'
