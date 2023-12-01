#!/bin/bash

# ./scripts/test_historical_batch.sh "ALAN" 48 "v1t v1s v1b" "2 3 4"
# ./scripts/test_historical_batch.sh "CARN DRRN EDSR-baseline LapSRN LatticeNet LBNet SwinIR" 48 "no" "2 3 4"
# ./scripts/test_historical_batch.sh "ShuffleMixer" 64 "base" "2 3 4"
# color is 1: ./scripts/test_historical_batch.sh "DRRN LapSRN" 48 "no" "2 3 4"

models=($1)
patch=$2
sizes=($3)
# scales=(2 3 4)
scales=($4)

for model in ${models[@]}; do 
#  echo "$model"
  for s in ${sizes[@]}; do 
  # echo "$s"
    for scale in ${scales[@]}; do 
        echo "./scripts/test_historical.sh $model $s $scale $patch"
        ./scripts/test_historical.sh $model $s $scale $patch
    done
  done
done

