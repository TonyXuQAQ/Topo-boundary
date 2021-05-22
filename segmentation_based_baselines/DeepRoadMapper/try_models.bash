python train_seg.py --mode infer_test
python ./candidate_test.py
python train_reason.py --mode test
python ./utils/eval_metric.py