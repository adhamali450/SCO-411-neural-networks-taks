class ConfusionMatrix:
    # contructor
    conf_data = {
        'true_pos': 0,
        'false_pos': 0,
        'false_neg': 0,
        'true_neg': 0
    }

    def __init__(self, y_true, y_predicted, true_label=1, false_label=0):
        if (len(y_true) != len(y_predicted)):
            raise Exception("y_true and y_predicted must have the same length")
        for _ in range(len(y_true)):
            if y_true == y_predicted == true_label:
                self.conf_data['true_pos'] += 1
            elif y_true == y_predicted == false_label:
                self.conf_data['true_neg'] += 1
            elif y_true == true_label and y_predicted == false_label:
                self.conf_data['false_neg'] += 1
            elif y_true == false_label and y_predicted == true_label:
                self.conf_data['false_pos'] += 1

    def print(self):
        print(f"True Positive: {self.conf_data['true_pos']}")
        print(f"False Positive: {self.conf_data['false_pos']}")
        print(f"False Negative: {self.conf_data['false_neg']}")
        print(f"True Negative: {self.conf_data['true_neg']}")

    def accuracy(self):
        # (TP + TN) / (TP + TN + FP + FN)
        return (self.conf_data['true_pos'] + self.conf_data['true_neg']) / (self.conf_data['true_pos'] + self.conf_data['true_neg'] + self.conf_data['false_pos'] + self.conf_data['false_neg'])

    def precision(self):
        # TP / (TP + FP)
        return self.conf_data['true_pos'] / (self.conf_data['true_pos'] + self.conf_data['false_pos'])

    def recall(self):
        # TP / (TP + FN)
        return self.conf_data['true_pos'] / (self.conf_data['true_pos'] + self.conf_data['false_neg'])
