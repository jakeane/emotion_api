from tensorflow.contrib.predictor import from_saved_model as load_model
from tensorflow.train import Example, Features, Feature, Int64List
from tokenization import FullTokenizer

class ApiModel:
    def __init__(self):
        self.THRESHOLD = 0.01
        
        self.LABELS_16 = [
            "afraid-terrified",
            "angry-furious",
            "sentimental-nostalgic",
            "proud-impressed",
            "prepared-confident",
            "anxious-apprehensive",
            "ashamed-embarrassed",
            "hopeful-anticipating",
            "faithful-trusting",
            "lonely-sad",
            "disappointed-devastated",
            "annoyed-disgusted",
            "guilty-jealous",
            "grateful-caring",
            "joyful-content",
            "excited-surprised"
        ]

        self.MAX_SEQ_LENGTH = 50

        self.tokenizer = FullTokenizer(
            vocab_file='vocab.txt', do_lower_case=True)

        self.model = load_model('model_data/model32')

    def predict(self, text: str):

        input_ids, input_mask, segment_ids, label_ids = self._convert_single_example(
            text)

        features: str = self._serialize_features(
            input_ids, input_mask, segment_ids, label_ids)

        probabilities = self.model({'examples': [features]})[
            "probabilities"][0]
        
        # Convert probabilities to one hot vector (probabilities.apply(lambda x: 1 if x >= 0.8 else 0))
        # Matrix multiply probabilites to map api_emotions.txt to animation_emotions.txt
        # Use Tensorflow matrix multiplication
        """
            return {
                "emotion": Whatever is 1 in probabilites.apply() vector. "neutral" otherwise
                "animations": vector from matrix multiplication
            }
        """
        prob_threshold = 0.8
        # value for "emotion" in return dict
        with open('api_emotions.txt', 'r') as f:
            api_emotions = [line.strip() for line in f.readlines()]
            
        excluded_emotions = ['nostalgic', 'sentimental', 'prepared', 'anticipating']
        emotions = [k for k,v in zip(api_emotions, probabilities) if (v>th) and (k not in excluded_emotions)] # recheck
        if len(emotions) == 0:
            emotions = ['neutral']
        
        # value for "animations" in return dict
        multiplier = np.genfromtxt('emotion_multiplier.csv')
        multiplier = tf.constant(multiplier, dtype='float32')
        temp = tf.constant(probabilities)
        temp = tf.map_fn(lambda x: tf.cond(tf.greater(x, prob_threshold), lambda: 1.0, lambda: 0.0), temp)
        temp = tf.matmul(multiplier, tf.reshape(temp, (-1, 1)))
        with tf.Session() as sess: 
            animations = tf.transpose(t3).eval()
        
        returning_dict = {'emotions': emotions, 'animations': animations}

        top_probabilities = [(k, v)
                             for k, v in zip(self.LABELS_16, probabilities)
                             if v >= self.THRESHOLD]

        return dict(sorted(top_probabilities, key=lambda x: -x[1]))

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

        return input_ids, input_mask, segment_ids, [0] * len(self.LABELS_16)

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
