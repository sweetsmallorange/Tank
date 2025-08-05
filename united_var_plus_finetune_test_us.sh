#tank模型第一次测试
SHELL_FOLDER=$(readlink -f "$0")

#注意test的时候dss_config_ds要单独设置了
#注意有没有finetune_name
python -u main.py \
  --sh_file "$SHELL_FOLDER" \
  --stage 'test' \
  --test_mode 'single' \
  --test_name 'us_50' \
  --finetune_name "us_all_pretrained" \
  --run_dir "/data2/zmz1/Tank/runs_paper/attr9_448CARTank_Parallel4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[CARTank,varFalse,batch_size2048]_[epochs200]_[15,1][2025062508]" \
  --model_id '448CARTank_Parallel4_runoff_Transformer' \
  --model 'CARTank' \
  --sub_model 'var_plus' \
  --seed 1234 \
  --camels_root '/data2/zmz1/Camels' \
  --forcing_type 'separated' \
  --basins_list_path '' \
  --loss 'mse' \
  --warm_up true \
  --epochs 200 \
  --past_len 15 \
  --pred_len 1 \
  --src_size 5 \
  --static_size 9 \
  --loss_all false \
  --use_var false \
  --gpu 1 \
  --learning_rate 0.001 \
  --dropout 0.1 \
  --drop_last false \
  --batch_size 2048 \
  --use_runoff true \
  --local_run_dir 'runs_united_runoff' \
  --freq 'd' \
  --dss_config_path '/data2/zmz1/Tank/data/SeriesTank_10basins_list.txt' \
  --dss_config_ds 'CAMELS-US'
