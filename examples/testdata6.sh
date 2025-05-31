#!/bin/bash
SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# "LinearModel" "KNN" "DecisionTree" "RandomForest"
# "XGBoost" "CatBoost" "LightGBM"
# "MLP" "TabNet" "VIME"
# MODELS=( "LinearModel" "KNN" "DecisionTree" "RandomForest" "XGBoost" "CatBoost" "LightGBM" "MLP" "TabNet" "VIME")

declare -A MODELS
MODELS=( #["mlp"]=$TORCH_ENV
         #['resnet']=$TORCH_ENV
         ["transformer"]=$TORCH_ENV
          )

SOURCE='rtdl-modelwise'

NUM_BINS_SOURCES=(
                  "model-lr"
                  #"model-t-lr"
                  #"model-lrlr"
                  #"model-t-lrlr"
                  )

# rtdl_datasets = {
#     'eye': eye_movements, # removed by rtdl in their paper
#     'gesture': gesture_phase,
#     'house': house_16h,
#     'higgs-small': higgs_small,
#     'santander': santander_customer_transactions,
#     'otto': otto_group_products,
#     'churn': churn_modelling,
#     'fb-comments': lambda: facebook_comments_volume(keep_derived=True),
#     'california': california_housing,
#     'covtype': covtype,
#     'adult': adult
# }
CONFIGS=( "config/rtdl/adult.yml"
          "config/rtdl/gesture.yml"
          "config/rtdl/house.yml"
          "config/rtdl/higgs-small.yml"
          "config/rtdl/otto.yml"
          "config/rtdl/santander.yml"
          "config/rtdl/churn.yml"
          "config/rtdl/fb-comments.yml"
          "config/rtdl/california.yml"
          "config/rtdl/covtype.yml"
          )

EMB_CONFIGS=( 
          #"config/model_default.yml"
          #"config/model_control_T.yml"
          #"config/model_control_R.yml"
          #"config/model_LR_standard.yml"          
          #"config/model_LR_alpha.yml"
          "config/model_LR_rtdl.yml"
          #"config/model_LT_standard.yml"
          #"config/model_LT_alpha.yml"
          
          #"config/model_LRLR_alpha.yml"
          #"config/model_LRLR_rtdl.yml"
          #"config/model_LRLR_standard.yml"
          #"config/model_LTLR_alpha.yml"
          #"config/model_LTLR_standard.yml"
          )



#TRAIN_CONFIG="config/config_debug.yml"
TRAIN_CONFIG="config/config_train.yml"

# conda init bash
eval "$(conda shell.bash hook)"

for NUM_BINS_SOURCE in "${NUM_BINS_SOURCES[@]}"; do

  for config in "${CONFIGS[@]}"; do
    
    for emb_config in "${EMB_CONFIGS[@]}"; do

      for model in "${!MODELS[@]}"; do
          
        printf "\n\n----------------------------------------------------------------------------\n"
        printf 'Training %s with %s in  %s, embedding config %s \n\n' "$model" "$config" "${MODELS[$model]}" "$emb_config"

        conda activate "${MODELS[$model]}"

        python train.py \
            --config "$config" \
            --train_config "$TRAIN_CONFIG" \
            --emb_config "$emb_config" \
            --model_name "$model" \
            --source "$SOURCE" \
            --num_bins_source "$NUM_BINS_SOURCE" \
            # --save_embedded_features \
            #--with_original_features \


        conda deactivate

      done

    done

  done

done