#!/bin/bash
EPOCHS=1 # number of epochs to run
BATCH_SIZE=1 # batch size to use
SAMPLE_RATE=16000 # sample rate of the audio
DATA_RAW=indic_splits # the unprocessed audio
DATA_PROCESSED=indic_rhythm_splits # 'rhythm' data
DATA_SUFFIX=_indic # a suffix so results are stored separately
DEVICE=cuda # device to run things on
RUN=1 # set to 0 to test the script without actually running anything

# Input sizes
AUDIO_INPUT_SIZE=4096
AUDIO_STRIDE=1024
SPEC_INPUT_SIZE=256
SPEC_STRIDE=128

# stats: unprocessed, raw
UNP_RAW_MEAN=0
UNP_RAW_STD=0.00219
# stats: unprocessed, features
UNP_MEL_MEAN=-14764.7
UNP_MEL_STD=53254518446
# stats: rhythm, raw
RHY_RAW_MEAN=0
RHY_RAW_STD=0.0012
# stats: rhythm, features
RHY_MEL_MEAN=-18337
RHY_MEL_STD=82141527591

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
CMD="$_CMD --data $DATA_RAW --features raw --combine MoT --model-id aud_raw_mot$DATA_SUFFIX --data-mean $UNP_RAW_MEAN --data-std $UNP_RAW_STD --input-size $AUDIO_INPUT_SIZE --input-stride $AUDIO_STRIDE"
run_cmd

# Experiment 2: Raw audio with CNN+LSTM
CMD="$_CMD --data $DATA_RAW --features raw --combine LSTM --model-id aud_raw_lstm$DATA_SUFFIX --data-mean $UNP_RAW_MEAN --data-std $UNP_RAW_STD --input-size $AUDIO_INPUT_SIZE --input-stride $AUDIO_STRIDE"
run_cmd

# Experiment 3: Raw audio features with CNN+MoT
CMD="$_CMD --data $DATA_RAW --features mel-spectogram  --combine MoT --model-id aud_ms_mot$DATA_SUFFIX --data-mean $UNP_MEL_MEAN --data-std $UNP_MEL_STD --input-size $SPEC_INPUT_SIZE --input-stride $SPEC_STRIDE"
run_cmd

# Experiment 4: Raw audio features with CNN+LSTM
CMD="$_CMD --data $DATA_RAW --features mel-spectogram  --combine LSTM --model-id aud_ms_lstm$DATA_SUFFIX --data-mean $UNP_MEL_MEAN --data-std $UNP_MEL_STD --input-size $SPEC_INPUT_SIZE --input-stride $SPEC_STRIDE"
run_cmd

# Experiment 5: Rhythm audio with CNN+MoT
CMD="$_CMD --data $DATA_PROCESSED --features raw --combine MoT --model-id rhy_raw_mot$DATA_SUFFIX --data-mean $RHY_RAW_MEAN --data-std $RHY_RAW_STD --input-size $AUDIO_INPUT_SIZE --input-stride $AUDIO_STRIDE"
run_cmd

# Experiment 6: Rhythm audio with CNN+LSTM
CMD="$_CMD --data $DATA_PROCESSED --features raw --combine LSTM --model-id rhy_raw_lstm$DATA_SUFFIX --data-mean $RHY_RAW_MEAN --data-std $RHY_RAW_STD --input-size $AUDIO_INPUT_SIZE --input-stride $AUDIO_STRIDE"
run_cmd

# Experiment 7: Rhythm audio features with CNN+MoT
CMD="$_CMD --data $DATA_PROCESSED --features mel-spectogram  --combine MoT --model-id rhy_ms_mot$DATA_SUFFIX --data-mean $RHY_MEL_MEAN --data-std $RHY_MEL_STD --input-size $SPEC_INPUT_SIZE --input-stride $SPEC_STRIDE"
run_cmd

# Experiment 8: Rhythm audio features with CNN+LSTM
CMD="$_CMD --data $DATA_PROCESSED --features mel-spectogram  --combine LSTM --model-id rhy_ms_lstm$DATA_SUFFIX --data-mean $RHY_MEL_MEAN --data-std $RHY_MEL_STD --input-size $SPEC_INPUT_SIZE --input-stride $SPEC_STRIDE"
run_cmd
