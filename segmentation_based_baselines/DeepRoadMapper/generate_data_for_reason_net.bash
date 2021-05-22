# infer the skeletons 
python train_seg.py --mode infer_train
python train_seg.py --mode infer_valid
python train_seg.py --mode infer_test

# generate connection candidates 
python ./candidate_train.py
python ./candidate_valid.py
python ./candidate_test.py