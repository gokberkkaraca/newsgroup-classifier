training_feature_vectors = []
with open('data/question-4-train-features.csv') as f:
    for line in f.read().splitlines():
        training_feature_vectors.append(line.split(','))

training_label_vectors = []
with open('data/question-4-train-labels.csv') as f:
    for line in f.read().splitlines():
        training_label_vectors.append(line)

test_feature_vectors = []
with open('data/question-4-test-features.csv') as f:
    for line in f.read().splitlines():
        test_feature_vectors.append(line.split(','))

test_label_vectors = []
with open('data/question-4-test-labels.csv') as f:
    for line in f.read().splitlines():
        test_label_vectors.append(line)