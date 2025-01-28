import React from "react";
import { View, Text, StyleSheet } from "react-native";

interface Prediction {
  label: string;
  probability: number;
}

interface HistoryItemProps {
  imageName: string;
  predictionResult: string; // JSON string
  timestamp: string;
}

export const HistoryItem: React.FC<HistoryItemProps> = ({
  imageName,
  predictionResult,
  timestamp,
}) => {
  // Parse and sort predictions
  const predictions: Prediction[] = JSON.parse(
    predictionResult
  ).prediction.sort(
    (a: Prediction, b: Prediction) => b.probability - a.probability
  );

  // Extract the most probable prediction and the remaining ones
  const topPrediction = predictions.length > 0 ? predictions[0] : null;
  const otherPredictions = predictions.slice(1);

  // Format other predictions into a single string
  const formattedOtherPredictions = otherPredictions
    .map(
      (pred: Prediction) =>
        `${pred.label} (${(pred.probability * 100).toFixed(1)}%)`
    )
    .join(", ");

  return (
    <View style={styles.historyItem}>
      <Text style={styles.historyText}>Image: {imageName}</Text>

      {/* Highlight the most probable prediction */}
      {topPrediction && (
        <Text style={styles.topPrediction}>
          Prediction: {topPrediction.label} (
          {(topPrediction.probability * 100).toFixed(1)}%)
        </Text>
      )}

      {/* Other predictions */}
      {formattedOtherPredictions.length > 0 && (
        <Text style={styles.historyText}>
          Other Classes: {formattedOtherPredictions}
        </Text>
      )}

      {/* Display the timestamp */}
      <Text style={styles.historyTimestamp}>
        Date: {new Date(timestamp).toLocaleString()}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  historyItem: {
    borderBottomWidth: 1,
    borderColor: "#ccc",
    paddingVertical: 10,
  },
  historyText: {
    fontSize: 16,
    color: "#333",
  },
  topPrediction: {
    fontSize: 16,
    fontWeight: "bold", // Make the top prediction bold
    color: "#008080", // Use a distinct color
    marginTop: 5,
  },
  historyTimestamp: {
    fontSize: 12,
    color: "gray",
    marginTop: 5,
  },
});

export default HistoryItem;
