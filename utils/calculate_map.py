from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt-file', type=str, default='./test_gt.json', help='Test GT path')
parser.add_argument('--pred-file', type=str, default='./test_pred.json', help='Test Pred path')
opt = parser.parse_args()


def coco_evaluation(gt_file, pred_file):
    # Initialize COCO ground truth API
    coco_gt = COCO(gt_file)

    # Load results in COCO prediction format
    with open(pred_file, 'r') as f:
        coco_pred = coco_gt.loadRes(json.load(f))

    #Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# File paths
gt_file = opt.gt_file
pred_file = opt.pred_file

# Evaluate
coco_evaluation(gt_file, pred_file)
