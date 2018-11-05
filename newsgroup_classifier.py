training_feature_vectors = []
with open('data/question-4-train-features.csv') as f:
    for line in f.read().splitlines():
        feature_vector = list(map(lambda x: int(x), line.split(',')))
        training_feature_vectors.append(feature_vector)

training_label_vectors = []
with open('data/question-4-train-labels.csv') as f:
    for line in f.read().splitlines():
        training_label_vectors.append(int(line))

test_feature_vectors = []
with open('data/question-4-test-features.csv') as f:
    for line in f.read().splitlines():
        feature_vector = list(map(lambda x: int(x), line.split(',')))
        test_feature_vectors.append(feature_vector)

test_label_vectors = []
with open('data/question-4-test-labels.csv') as f:
    for line in f.read().splitlines():
        test_label_vectors.append(int(line))

prob_class_is_0 = len(list(filter(lambda x: x == 0, training_label_vectors))) / len(training_label_vectors)
prob_class_is_1 = len(list(filter(lambda x: x == 1, training_label_vectors))) / len(training_label_vectors)

print(prob_class_is_0, prob_class_is_1)
