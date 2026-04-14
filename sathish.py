                    # Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("=== MODEL EVALUATION ===")
print(f"Accuracy Score: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived']
            }
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('chart_confusion_matrix.png')
plt.show()
print("Confusion Matrix saved!")
