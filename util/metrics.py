# code adapted, original source: ISEAD project (HITeC e.V.)

import csv


def read_txt(file_path):
    test = []
    with open(file_path, "r") as fp:
        for line in fp.readlines():
            test.append(line.removesuffix("\n"))
    return test


# read prediction csv file
def read_csv(file_path, test_file, delimiter=";"):
    data = []
    # test split
    if test_file:
        # get test data filenames
        test = read_txt(test_file)
    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        try:
            headers = next(csv_reader)
        except StopIteration:
            print("No lines in csv")

        for row in csv_reader:
            # filter test data if necessary
            if (
                test_file
                and (((row[0].removesuffix(".jpg")).removesuffix(".png"))) in test
            ) or not test_file:
                if float(row[7]) > 0.1:
                    data.append(
                        {
                            headers[i]: (
                                ((row[i].removesuffix(".jpg")).removesuffix(".png"))
                            )
                            for i in range(len(headers))
                        }
                    )
    return data


def calculate_iou(bbox1, bbox2):
    x1 = max(float(bbox1["xmin"]), float(bbox2["xmin"]))
    y1 = max(float(bbox1["ymin"]), float(bbox2["ymin"]))
    x2 = min(float(bbox1["xmax"]), float(bbox2["xmax"]))
    y2 = min(float(bbox1["ymax"]), float(bbox2["ymax"]))

    intersection_area = max(0, x2 - x1 + 0.0001) * max(0, y2 - y1 + 0.0001)
    bbox1_area = (float(bbox1["xmax"]) - float(bbox1["xmin"]) + 0.0001) * (
        float(bbox1["ymax"]) - float(bbox1["ymin"]) + 0.0001
    )
    bbox2_area = (float(bbox2["xmax"]) - float(bbox2["xmin"]) + 0.0001) * (
        float(bbox2["ymax"]) - float(bbox2["ymin"]) + 0.0001
    )
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def calculate_metrics_bbox_level(ground_truth, predictions, iou_threshold):
    true_positive_predictions = []
    false_positive_predictions = []
    for pred in predictions:
        false_positive_predictions.append(pred)
    false_negative_predictions = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for gt_bbox in ground_truth:
        found_matching_prediction = False
        for pred_bbox in predictions:
            # only if image is also the same
            if pred_bbox["image"] == gt_bbox["image"]:
                iou = calculate_iou(gt_bbox, pred_bbox)
                if iou >= iou_threshold:
                    true_positives += 1
                    true_positive_predictions.append(gt_bbox)
                    if pred_bbox in false_positive_predictions:
                        false_positive_predictions.remove(pred_bbox)
                    found_matching_prediction = True
                    break
        if not found_matching_prediction:
            false_negative_predictions.append(gt_bbox)
            false_negatives += 1

    false_positives = len(false_positive_predictions)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return (precision, recall, f1_score)


def bbox_metrics(predictions_seg_file, ground_truth_test_file_csv, val_file):

    iou_threshold = 0.005
    # read data from csvs
    ground_truth_seg = read_csv(ground_truth_test_file_csv, val_file)
    predictions_seg = read_csv(predictions_seg_file, val_file)
    # calculate metrics for Segmentation
    metrics = calculate_metrics_bbox_level(
        ground_truth_seg, predictions_seg, iou_threshold
    )
    return metrics
