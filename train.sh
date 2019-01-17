EPOCHS=1
BATCH_SIZE=2
SAMPLE_RATE=8000
DATA_RAW=new_dataset
DATA_PROCESSED=new_dataset_processed
DEVICE=cpu


_CMD="python rhythm.py train --sample-rate $SAMPLE_RATE  --batch-size $BATCH_SIZE --epochs $EPOCHS --device $DEVICE"

# Experiment 1: Raw audio with CNN+MoT 
CMD="$_CMD --data $DATA_RAW --features raw  --combine MoT --model-id aud_raw_mot"
echo $CMD; 
eval $CMD;
echo $CMD --test; 
eval $CMD --test;

# Experiment 2: Raw audio with CNN+LSTM

# Experiment 3: Raw audio features with CNN+MoT
CMD="$_CMD --data $DATA_RAW --features mel-spectogram  --combine MoT --model-id aud_ms_mot"
echo $CMD; 
eval $CMD;
echo $CMD --test; 
eval $CMD --test;

# Experiment 4: Raw audio features with CNN+LSTM 
# Experiment 5: Rhythm audio with CNN+MoT 
CMD="$_CMD --data $DATA_PROCESSED --features raw  --combine MoT --model-id rhy_raw_mot"
echo $CMD; 
eval $CMD;
echo $CMD --test; 
eval $CMD --test;
# Experiment 6: Rhythm audio with CNN+LSTM 
# Experiment 7: Rhythm audio features with CNN+MoT
CMD="$_CMD --data $DATA_PROCESSED --features mel-spectogram  --combine MoT --model-id rhy_ms_mot"
echo $CMD; 
eval $CMD;
echo $CMD --test; 
eval $CMD --test;

# Experiment 8: Rhythm audio features with CNN+LSTM
