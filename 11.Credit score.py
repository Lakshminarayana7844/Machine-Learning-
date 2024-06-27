def categorize_credit_score(score):
    if score < 580:
        return 'Poor'
    elif score < 670:
        return 'Fair'
    elif score < 740:
        return 'Good'
    else:
        return 'Excellent'

# Example credit scores
credit_scores = [500, 600, 650, 700, 750, 800]

# Classify each credit score
classified_scores = [categorize_credit_score(score) for score in credit_scores]

# Print the results
for score, category in zip(credit_scores, classified_scores):
    print(f"Credit Score: {score} -> Category: {category}")
