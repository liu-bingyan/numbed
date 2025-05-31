#!/bin/bash
SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# "LinearModel" "KNN" "DecisionTree" "RandomForest"
# "XGBoost" "CatBoost" "LightGBM"
# "MLP" "TabNet" "VIME"
# MODELS=( "LinearModel" "KNN" "DecisionTree" "RandomForest" "XGBoost" "CatBoost" "LightGBM" "MLP" "TabNet" "VIME")

SOURCE='TabSurvey'

declare -A MODELS
MODELS=( #["LinearModel"]=$SKLEARN_ENV
         #["KNN"]=$SKLEARN_ENV
         #["SVM"]=$SKLEARN_ENV
         #["DecisionTree"]=$SKLEARN_ENV
         #["RandomForest"]=$SKLEARN_ENV
         #["XGBoost"]=$GBDT_ENV
         #["CatBoost"]=$GBDT_ENV
         #["LightGBM"]=$GBDT_ENV
         
         ["MLP"]=$TORCH_ENV
         #['ResNet']=$TORCH_ENV
         #["TabNet"]=$TORCH_ENV
         #["VIME"]=$TORCH_ENV
         #["TabTransformer"]=$TORCH_ENV
         #["ModelTree"]=$GBDT_ENV
         #["NODE"]=$TORCH_ENV
         #["DeepGBM"]=$TORCH_ENV
         #["RLN"]=$KERAS_ENV
         #["DNFNet"]=$KERAS_ENV
         #["STG"]=$TORCH_ENV
         #["NAM"]=$TORCH_ENV
         #["DeepFM"]=$TORCH_ENV
         #["SAINT"]=$TORCH_ENV
         #["DANet"]=$TORCH_ENV
          )

CONFIGS=( #"config/TabSurvey/adult.yml"
           "config/TabSurvey/california_housing.yml"
          #"config/TabSurvey/covertype.yml"
          #"config/TabSurvey/higgs.yml"
          )


EMB_CONFIGS=( 
          "config/model_default.yml"
          "config/model_control_T.yml"
          #"config/model_control_R.yml"
          #"config/model_LR_standard.yml"          
          #"config/model_LR_alpha.yml"
          #"config/model_LT_standard.yml"
          "config/model_LT_alpha.yml"
          #"config/model_LRLR_alpha.yml"
          #"config/model_LRLR_standard.yml"
          #"config/model_LTLR_alpha.yml"c
          #"config/model_LTLR_standard.yml"
          )

NUM_BINS=17
TRAIN_CONFIG="config/config_debug.yml"

# conda init bash
eval "$(conda shell.bash hook)"

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
          --num_bins $NUM_BINS \
          --source "$SOURCE" \
          #--with_original_features \
          #--save_embedded_features \


      conda deactivate

    done

  done

done
