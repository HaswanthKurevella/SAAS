import matplotlib.pyplot as plt
import numpy as np

# Use accuracy values from the previous computation
accuracies = [gaze_accuracy, emotion_accuracy, integrated_accuracy]

# Define labels and colors
labels = ['Gaze Detection', 'Emotion Recognition', 'Integrated System']
colors = ['blue', 'green', 'red']

# Scale values for better visibility (optional)
scaled_accuracies = [acc * 2 for acc in accuracies]

# Create the bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, scaled_accuracies, color=colors)

# Add value labels showing the actual accuracy
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, 
             bar.get_height() + 0.02, 
             f"{acc:.2f}",  # Display original accuracy
             fontsize=12, fontweight='bold')

# Adjust y-axis for better visibility
plt.ylim(0, max(scaled_accuracies) + 0.2)
plt.ylabel("Scaled Accuracy (Original values shown on bars)")
plt.title("Enhanced Comparison of Engagement Detection Methods")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.savefig("engagement_detection_comparison.png", dpi=300, bbox_inches='tight')  # Save as PNG
# plt.savefig("engagement_detection_comparison.pdf")  # Save as PDF
# plt.savefig("engagement_detection_comparison.jpg", dpi=300)  # Save as JPG

# Show the plot
plt.show()
