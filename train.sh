#!/bin/bash
EPOCHS=1
BATCH_SIZE=1
SAMPLE_RATE=16000
DATA_RAW=indic_splits
DATA_PROCESSED=indic_rhythm_splits
DATA_SUFFIX=_indic
DEVICE=cpu
RUN=0

# stats: unprocessed, raw
UNP_RAW_MEAN=URM
UNP_RAW_STD=URS
# stats: unprocessed, features
UNP_MEL_MEAN=UMM
UNP_MEL_STD=UMS
# stats: rhythm, raw
RHY_RAW_MEAN=RRM
RHY_RAW_STD=RRS
# stats: rhythm, features
RHY_MEL_MEAN=RMM
RHY_MEL_STD=RMS

run_cmd() {
    echo $CMD;
    if [ $RUN == 1 ];
    then 
        eval $CMD;
    fi

    #echo $CMD --test ;
    if [ $RUN == 1 ];
    then 
        eval $CMD;
    fi 
}


_CMD="python rhythm.py train --sample-rate $SAMPLE_RATE  --batch-size $BATCH_SIZE --epochs $EPOCHS --device $DEVICE "

# Experiment 1: Raw audio with CNN+MoT 
CMD="$_CMD --data $DATA_RAW --features raw --combine MoT --model-id aud_raw_mot$DATA_SUFFIX --data-mean $UNP_RAW_MEAN --data-std $UNP_RAW_STD"
run_cmd

# Experiment 2: Raw audio with CNN+LSTM
CMD="$_CMD --data $DATA_RAW --features raw --combine LSTM --model-id aud_raw_lstm$DATA_SUFFIX --data-mean $UNP_RAW_MEAN --data-std $UNP_RAW_STD"
run_cmd

# Experiment 3: Raw audio features with CNN+MoT
CMD="$_CMD --data $DATA_RAW --features mel-spectogram  --combine MoT --model-id aud_ms_mot$DATA_SUFFIX --data-mean $UNP_MEL_MEAN --data-std $UNP_MEL_STD"
run_cmd

# Experiment 4: Raw audio features with CNN+LSTM 
CMD="$_CMD --data $DATA_RAW --features mel-spectogram  --combine LSTM --model-id aud_ms_lstm$DATA_SUFFIX --data-mean $UNP_MEL_MEAN --data-std $UNP_MEL_STD"
run_cmd

# Experiment 5: Rhythm audio with CNN+MoT 
CMD="$_CMD --data $DATA_PROCESSED --features raw --combine MoT --model-id rhy_raw_mot$DATA_SUFFIX --data-mean $RHY_RAW_MEAN --data-std $RHY_RAW_STD"
run_cmd

# Experiment 6: Rhythm audio with CNN+LSTM 
CMD="$_CMD --data $DATA_PROCESSED --features raw --combine LSTM --model-id rhy_raw_lstm$DATA_SUFFIX --data-mean $RHY_RAW_MEAN --data-std $RHY_RAW_STD"
run_cmd

# Experiment 7: Rhythm audio features with CNN+MoT
CMD="$_CMD --data $DATA_PROCESSED --features mel-spectogram  --combine MoT --model-id rhy_ms_mot$DATA_SUFFIX --data-mean $RHY_MEL_MEAN --data-std $RHY_MEL_STD"
run_cmd

# Experiment 8: Rhythm audio features with CNN+LSTM
CMD="$_CMD --data $DATA_PROCESSED --features mel-spectogram  --combine LSTM --model-id rhy_ms_lstm$DATA_SUFFIX --data-mean $RHY_MEL_MEAN --data-std $RHY_MEL_STD"
run_cmd
