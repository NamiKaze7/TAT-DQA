#!/usr/bin/python

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, Optional
import json
import argparse
from pathlib import Path
import numpy as np
from tatqa_utils import *
from tatqa_metric import TaTQAEmAndF1
from tatqa_metric import *
import pandas as pd


# From here through _normalize_answer was originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up and modified a bit.
def evaluate_json(annotations: Dict[str, Any], predicted_answers: Dict[str, Any], mode=Mode.NUMBER_ONLY) -> Tuple[float, float]:

    em_and_f1 = TaTQAEmAndF1(mode=mode)
    for qas in annotations:
        for qa in qas["questions"]:
            query_id = qa["uid"]
            pred_answer, pred_scale, pred_type = None, None, None
            if query_id in predicted_answers:
                pred_answer, pred_scale, pred_type = predicted_answers[query_id]
            em_and_f1(ground_truth=qa, prediction=pred_answer, pred_type=pred_type, pred_scale=pred_scale)

    global_em, global_f1, global_scale = em_and_f1.get_overall_metric()
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("scale score {0:.2f}".format(global_scale * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")

    detail_raw = em_and_f1.get_raw_pivot_table()
    print("---- raw detail ---")
    print(detail_raw)
    detail_em, detail_f1 = em_and_f1.get_detail_metric()
    print("---- em detail ---")
    print(detail_em)
    print("---- f1 detail ---")
    print(detail_f1)
    raws = em_and_f1.get_raw()
    detail_em, detail_f1 = em_and_f1.get_detail_metric()
    raw_detail = em_and_f1.get_raw_pivot_table()
    detail_em.to_excel("NumNet_test_detail_em.xls")
    detail_f1.to_excel("NumNet_test_detail_f1.xls")
    raws = pd.DataFrame.from_records(raws)
    raws.to_excel("NumNet_test_raws.xls")
    raw_detail.to_excel("NumNet_test_raw_detail.xls")

    return global_em, global_f1, detail_raw, detail_em, detail_f1


def evaluate_prediction_file(prediction_path: str,
                             gold_path: str,
                             output_folder: Optional[str] = None,
                             mode=Mode.NUMBER_ONLY):

    predicted_answers = json.load(open(prediction_path, encoding='utf-8'))
    annotations = json.load(open(gold_path, encoding='utf-8'))
    global_em, global_f1, detail_raw, detail_em, detail_f1 = evaluate_json(annotations, predicted_answers, mode)

    # Output predictions to file if an output path is given
    # if output_folder is not None:
    #     global_metric = {"global_em": global_em,
    #                    "global_f1": global_f1}
    #
    #     Path(output_folder).mkdir(exist_ok=True, parents=True)
    #     with Path(output_folder).joinpath('global_metric.json').open('w') as outfile:
    #         json.dump(global_metric, outfile)
    #
    #     with Path(output_folder).joinpath('detail_raw.xlsx').open('w') as outfile:
    #         detail_raw.to_excel(outfile, index=False)
    #
    #     with Path(output_folder).joinpath('detail_em.xlsx').open('w') as outfile:
    #         detail_em.to_excel(outfile, index=False)
    #
    #     with Path(output_folder).joinpath('detail_f1.xlsx').open('w') as outfile:
    #         detail_f1.to_excel(outfile, index=False)


if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='evaluation on tatqa dataset')
    parser.add_argument("--gold_path",
                        type=str,
                        required=True,
                        default="drop_dataset_test.gold.json",
                        help='location of the gold file')
    parser.add_argument("--prediction_path",
                        type=str,
                        required=True,
                        default="sample_predictions.json",
                        help='location of the prediction file')
    parser.add_argument("--output_folder",
                        type=str,
                        required=False,
                        default=None,
                        help='location of the output metrics folder')
    parser.add_argument("--mode",
                        type=int,
                        required=False,
                        default=Mode.NUMBER_ONLY,
                        help='the evaluation mode, 1: number only, 2: number and scale')

    args = parser.parse_args()
    evaluate_prediction_file(args.prediction_path, args.gold_path, args.output_folder, args.mode)
