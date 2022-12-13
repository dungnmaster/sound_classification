import logging
from pyAudioAnalysis.audioSegmentation import mid_term_file_classification, labels_to_segments
from pyAudioAnalysis.audioTrainTest import load_model
from pyAudioAnalysis.audioTrainTest import extract_features_and_train
from db_writer import DataSink
from datetime import datetime

logging.basicConfig(
    filename='classifier.log',
    filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
    level=logging.DEBUG
)


class AudioClassifier:

    def __init__(self, trained_model):
        self.trained_model = trained_model
        self.db = DataSink()

    def classify(self, filepath):
        labels, class_names, _, _ = mid_term_file_classification(
            filepath, self.trained_model, "svm_rbf", True,
        )
        starttime = filepath.split('/')[-1].split('.')[0]
        if not starttime.isnumeric():
            logging.error("segment file name should be start timestamp...offset calculation will FAIL")
        starttime = int(starttime) if starttime.isnumeric() else 0

        logging.info("\nFix-sized segments:")
        for il, l in enumerate(labels):
            print(f'fix-sized segment {il}: {class_names[int(l)]}')

        # load the parameters of the model (actually we just want the mt_step here):
        cl, m, s, m_classes, mt_win, mt_step, s_win, s_step, c_beat = load_model(self.trained_model)

        # print "merged" segments (use labels_to_segments())
        logging.info("\nSegments:")
        segs, c = labels_to_segments(labels, mt_step)

        write_records = []
        for iS, seg in enumerate(segs):
            s_timestamp = datetime.fromtimestamp(starttime+int(seg[0]), tz=None) 
            e_timestamp = datetime.fromtimestamp(starttime+int(seg[1]), tz=None) 
            duration = int(seg[1]) - int(seg[0])
            label = class_names[int(c[iS])]
            write_records.append([s_timestamp, e_timestamp, label, duration])
            logging.info(f'segment {iS} {seg[0]} sec - {seg[1]} sec: {class_names[int(c[iS])]}')
        self.db.insert(write_records)

def train_model():
    mt, st = 1.0, 0.05
    dirs = [
        "datasets/Training_Dataset/Toilet",
        "datasets/raspberry-pi/Sink",
        "datasets/raspberry-pi/Shower",
        "datasets/raspberry-pi/Silence",
    ]
    extract_features_and_train(dirs, mt, mt, st, st, "svm_rbf", "model/audio_classifier")


if __name__ == '__main__':
    train_model()
