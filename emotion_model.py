from tensorflow.contrib.predictor import from_saved_model as load_model
from tensorflow.train import Example, Features, Feature, Int64List
from tokenization import FullTokenizer
import numpy as np

class ApiModel:
    def __init__(self):
        self.THRESHOLD = 0.1
        self.PROB_THRESHOLD = 0.8
        
        self.LABELS_32 = [
            "sentimental",
            "afraid",
            "proud",
            "faithful",
            "terrified",
            "joyful",
            "angry",
            "sad",
            "jealous",
            "grateful",
            "prepared",
            "embarrassed",
            "excited",
            "annoyed",
            "lonely",
            "ashamed",
            "guilty",
            "surprised",
            "nostalgic",
            "confident",
            "furious",
            "disappointed",
            "caring",
            "trusting",
            "disgusted",
            "anticipating",
            "anxious",
            "hopeful",
            "content",
            "impressed",
            "apprehensive",
            "devastated"
        ]

        self.MAX_SEQ_LENGTH = 50

        self.tokenizer = FullTokenizer(
            vocab_file='vocab.txt', do_lower_case=True)

        self.model = load_model('model_data/model32')

        self.matrix = np.genfromtxt('emotion_multiplier.csv')

        self.map_probabilities = np.vectorize(lambda x: 1 if x >= self.PROB_THRESHOLD else 0)

    def predict(self, text: str):

        input_ids, input_mask, segment_ids, label_ids = self._convert_single_example(
            text)

        features: str = self._serialize_features(
            input_ids, input_mask, segment_ids, label_ids)

        probabilities = self.model({'examples': [features]})[
            "probabilities"][0]
        
        # excluded_emotions = ['nostalgic', 'sentimental', 'prepared', 'anticipating']
        # emotions = [k for k,v in zip(self.LABELS_32, probabilities) if (v>self.PROB_THRESHOLD) and (k not in excluded_emotions)] # recheck
        # if len(emotions) == 0:
        #     emotions = ['neutral']

        animations = list(np.matmul(self.matrix, self.map_probabilities(probabilities)))

        top_probabilities = [(k, v)
                             for k, v in zip(self.LABELS_32, probabilities)
                             if v >= self.THRESHOLD]
        
        top_emotions = dict(sorted(top_probabilities, key=lambda x: -x[1]))

        return {'emotions': top_emotions, 'animations': animations}


    def _convert_single_example(self, text):
        """Modified from goemotions/bert_classifier.py"""
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) > self.MAX_SEQ_LENGTH - 2:
            tokens = tokens[0:(self.MAX_SEQ_LENGTH - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.MAX_SEQ_LENGTH:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        return input_ids, input_mask, segment_ids, [0] * len(self.LABELS_32)

    def _serialize_features(self, input_ids, input_mask, segment_ids, label_ids):
        features = {
            "input_ids": self._create_int_feature(input_ids),
            "input_mask": self._create_int_feature(input_mask),
            "segment_ids": self._create_int_feature(segment_ids),
            "label_ids": self._create_int_feature(label_ids)
        }

        tf_example = Example(features=Features(feature=features))

        return tf_example.SerializeToString()

    def _create_int_feature(self, values):
        return Feature(int64_list=Int64List(value=list(values)))


if __name__ == "__main__":
    print("Loading model")

    apimodel = ApiModel()
    print("making prediction")
    apimodel.predict(
        "sometimes I feel like I have no control over my life")

else:
    apimodel = ApiModel()


def get_model():
    return apimodel
