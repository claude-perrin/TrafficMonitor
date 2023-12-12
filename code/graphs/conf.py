CLASSES_TO_IDX = {"background": 0, "bus": 1, "cyclist": 2, "pedestrian":3, "car": 4}
IDX_TO_CLASSES = {0: "background", 1: "bus", 2: "cyclist", 3: "pedestrian", 4: "car"}
COLOR_DICT = {
    'Red': (255, 0, 0),
    'Green': (0, 255, 0),
    'Yellow': (255, 255, 0),
    'Purple': (128, 0, 128)
}
BOX_COLOR = {class_logit: color for class_logit, color in zip(list(CLASSES_TO_IDX.values())[1:], COLOR_DICT.values())}
NUMBER_OF_CLASSES = len(CLASSES_TO_IDX)
ROAD_ROI_POLYGON= [[0.04038564, 0.19765616],
       [0.16543429, 0.40349086],
       [0.06146425, 0.45163814],
       [0.08342231, 0.58491565],
       [0.34084437, 0.65749477],
       [0.54624344, 1.        ],
       [0.99936986, 0.99990168],
       [1.        , 0.73741499],
       [0.10464401, 0.18617171]]

# ROAD_ROI_POLYGON = [[0.2011398 , 0.22757663],
#        [0.26722889, 0.42399546],
#        [0.18537682, 0.44088946],
#        [0.18329949, 0.61986711],
#        [0.33146482, 0.65675885],
#        [0.46431244, 0.99853825],
#        [1.        , 1.        ],
#        [1.        , 0.83915979],
#        [0.28004125, 0.23212712]]

