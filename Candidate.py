import pandas as pd

def candidate_elimination(examples):
    specific_h = examples.iloc[0, :-1].values  # Initialize the most specific hypothesis
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]  # Initialize the most general hypothesis

    for index, example in examples.iterrows():
        if example.iloc[-1] == 'Y':  # If the example is positive
            for i in range(len(specific_h)):
                if example.iloc[i] != specific_h[i]:
                    specific_h[i] = '?'  # Generalize specific hypothesis
                    general_h[i][i] = '?'  # Generalize general hypothesis
        else: 
            for i in range(len(specific_h)):
                if example.iloc[i] != specific_h[i]:
                    general_h[i][i] = specific_h[i]  # Specialize general hypothesis

    # Remove incomplete rows from general_hypothesis
    general_h = [h for h in general_h if h != ['?' for _ in range(len(specific_h))]]

    return specific_h, general_h

# Example usage:
if __name__ == '__main__':
    # Read the dataset from CSV
    file_path = '/Users/lakshminarayanamandi/Downloads/Movies/ML/6.csv'  
    data = pd.read_csv(file_path)

    specific, general = candidate_elimination(data)

    print("Specific hypothesis:", specific)
    print("General hypotheses:")
    for hypothesis in general:
        print(hypothesis)
