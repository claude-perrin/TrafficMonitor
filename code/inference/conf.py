# CLASSES_TO_IDX = {"background": 0, "car": 1, "pedestrian": 2}
# IDX_TO_CLASSES = {0: "background", 1: "car", 2: "pedestrian"}
CLASSES_TO_IDX = {"background": 0, "bus": 1, "cyclist": 2, "pedestrian":3, "car": 4}
IDX_TO_CLASSES = {0: "background", 1: "bus", 2: "cyclist", 3: "pedestrian", 4: "car"}

COLOR_DICT = {
    'Green': (128, 0, 128),
    'Red': (128, 0, 128),
    'Yellow': (128, 0, 128),
    'Purple': (128, 0, 128),


}
BOX_COLOR = {class_logit: color for class_logit, color in zip(list(CLASSES_TO_IDX.values())[1:], COLOR_DICT.values())}
NUMBER_OF_CLASSES = len(CLASSES_TO_IDX)
ROAD_ROI_POLYGON = normalized_new_coordinates = [
    (0.327479588842014, 0.11983471074380166),
    (0.4276860060357219, 0.3994490359976714),
    (0.3822323624586777, 0.44403769841269845),
    (0.4349161053712766, 0.7134986225895317),
    (0.605371276613536, 0.7906337216868474),
    (0.6830109586131814, 0.9944891160770835),
    (0.9975258263300311, 0.993113008668916),
    (0.9989671921487604, 0.6074380165289256),
    (0.39095274278996886, 0.10606060606060606)
]


