CLASSES_TO_IDX = {"background": 0, "bus": 1, "cyclist": 2, "pedestrian":3, "vehicle": 4}
IDX_TO_CLASSES = {0: "background", 1: "bus", 2: "cyclist", 3: "pedestrian", 4: "vehicle"}
COLOR_DICT = {
    'Red': (255, 0, 0),
    'Green': (0, 255, 0),
    'Yellow': (255, 255, 0),
    'Purple': (128, 0, 128)
}
BOX_COLOR = {class_logit: color for class_logit, color in zip(list(CLASSES_TO_IDX.values())[1:], COLOR_DICT.values())}
NUMBER_OF_CLASSES = len(CLASSES_TO_IDX)
ROAD_ROI_POLYGON = [(0.9993742177712716, 0.9983221476510067),
                        (0.35469211514440676, 1.0),
                        (0.24616996245206507, 0.4462402661396575),
                        (0.24043429286608384, 0.32073429536223916),
                        (0.2943324155193992, 0.3233532934131736),
                        (0.9993742177712716, 0.9332809629834899)]
