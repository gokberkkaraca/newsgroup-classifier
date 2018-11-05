from math import log, log2, inf

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

# Train the model
prob_class_is_0 = len(list(filter(lambda x: x == 0, training_label_vectors))) / len(training_label_vectors)
prob_class_is_1 = len(list(filter(lambda x: x == 1, training_label_vectors))) / len(training_label_vectors)

print(prob_class_is_0, prob_class_is_1)

set_0 = []
set_1 = []
for i in range(len(training_feature_vectors)):
    if training_label_vectors[i] == 0:
        set_0.append(training_feature_vectors[i])
    else:
        set_1.append(training_feature_vectors[i])

vocab_size = 26507

qjy0 = []
sum_tjy0 = 0
for word in range(vocab_size):
    tjy0 = 0
    for email in set_0:
        tjy0 += email[word]
    qjy0.append(tjy0)
    sum_tjy0 += tjy0
qjy0 = list(map(lambda tjy0: (tjy0 + 1)/(sum_tjy0 + vocab_size), qjy0))

qjy1 = []
sum_tjy1 = 0
for word in range(vocab_size):
    tjy1 = 0
    for email in set_1:
        tjy1 += email[word]
    qjy1.append(tjy1)
    sum_tjy1 += tjy1
qjy1 = list(map(lambda tjy1: (tjy1 + 1)/(sum_tjy1 + vocab_size), qjy1))

# Test data
test_prediction_results = []
for document in test_feature_vectors:
    likelihood_0 = 0
    likelihood_1 = 0

    for word in range(vocab_size):
        if document[word] == 0:
            likelihood_0 += 0
            likelihood_1 += 0
        else:
            if qjy0[word] == 0:
                likelihood_0 += -inf
            else:
                likelihood_0 += document[word] * log(qjy0[word])
            
            if qjy1[word] == 0:
                likelihood_1 += -inf
            else:
                likelihood_1 += document[word] * log(qjy1[word])

    belongs_to_0 = log(prob_class_is_0) + likelihood_0
    belongs_to_1 = log(prob_class_is_1) + likelihood_1

    if belongs_to_1 > belongs_to_0:
        test_prediction_results.append(1)
    else:
        test_prediction_results.append(0)

result = list(zip(test_prediction_results, test_label_vectors))
result = list(map(lambda x: 1 if x[0] == x[1] else 0, result))
accuracy = sum(result) / len(result)
print(accuracy)

index_mi_score_tuples = []
for t in range(vocab_size):
    for c in [0,1]:
        
        n11 = len(list(filter(lambda x: x[t] != 0 and training_label_vectors[t] == c, training_feature_vectors)))
        n10 = len(list(filter(lambda x: x[t] != 0 and training_label_vectors[t] != c, training_feature_vectors)))
        n01 = len(list(filter(lambda x: x[t] == 0 and training_label_vectors[t] == c, training_feature_vectors)))
        n00 = len(list(filter(lambda x: x[t] == 0 and training_label_vectors[t] != c, training_feature_vectors)))
        
        print(n11, n10, n01, n00)

        n1dot = n10 + n11
        ndot1 = n01 + n11
        n0dot = n01 + n00
        ndot0 = n10 + n00

        n = n00 + n01 + n10 + n11

        mi_term1 = n11/n * log2(n*n11/(n1dot*ndot1))
        mi_term2 = n01/n * log2(n*n01/(n0dot*ndot1))
        mi_term3 = n10/n * log2(n*n10/(n1dot*ndot0))
        mi_term4 = n00/n * log2(n*n00/(n0dot*ndot0))

        mi_score = mi_term1 + mi_term2 + mi_term3 + mi_term4
        index_mi_score_tuples.append(t, mi_score)