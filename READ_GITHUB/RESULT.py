import pandas as pd

# Load results
results = pd.read_csv('blink_detection_results.csv')

# Count true positives and true blinks
detected_blinks = len(results[results['Detected Blink'] == 'True Blink'])
true_blinks = len(results[results['True Blink'] == 'True Blink'])

# Calculate accuracy
if true_blinks > 0:
    accuracy = (detected_blinks / true_blinks) * 100
else:
    accuracy = 0

print(f"Accuracy: {accuracy:.2f}%")
