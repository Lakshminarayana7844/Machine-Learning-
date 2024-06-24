def find_s_algorithm(training_data):
    h = ['0'] * len(training_data[0][0])
    for instance in training_data:
        attributes, label = instance
        if label == 'Yes': 
            for i in range(len(h)):
                if h[i] == '0':
                    h[i] = attributes[i]
                elif h[i] != attributes[i]: 
                    h[i] = '?'
    return h
training_data = [
    (['Sunny', 'Warm', 'Normal', 'Strong'], 'Yes'),
    (['Sunny', 'Warm', 'High', 'Strong'], 'Yes'),
    (['Rainy', 'Cold', 'High', 'Strong'], 'No'),
]
most_specific_hypothesis = find_s_algorithm(training_data)
print("The most specific hypothesis found by FIND-S is:", most_specific_hypothesis)
