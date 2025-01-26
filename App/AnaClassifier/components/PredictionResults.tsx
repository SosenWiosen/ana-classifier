import React from "react";
import { View, Text, StyleSheet } from "react-native";

export type PredictionItem = {
  label: string;
  probability: number;
};

type PredictionResultsProps = {
  predictions: PredictionItem[];
};

export const PredictionResults: React.FC<PredictionResultsProps> = ({
  predictions,
}) => {
  if (!predictions || predictions.length === 0) {
    return <Text style={styles.emptyText}>No predictions to display.</Text>;
  }
  console.log("Prediction Type:", typeof predictions);
  console.log("Is Prediction an Array?", Array.isArray(predictions));
  console.log("Prediction Contents:", predictions);
  return (
    <View style={styles.predictionContainer}>
      <Text style={styles.predictionTitle}>Prediction Results</Text>
      {predictions
        .sort((a, b) => b.probability - a.probability) // Sort by probability descending
        .map((item, index) => (
          <View
            key={index}
            style={[
              styles.predictionItem,
              index === 0 && styles.topPrediction, // Highlight top prediction
            ]}
          >
            <Text style={styles.predictionLabel}>{item.label}</Text>
            <Text style={styles.predictionProbability}>
              Confidence: {(item.probability * 100).toFixed(2)}%
            </Text>
          </View>
        ))}
    </View>
  );
};

const styles = StyleSheet.create({
  predictionContainer: {
    marginVertical: 20,
    padding: 10,
    width: "80%",
    alignSelf: "center",
    borderColor: "#ddd",
    borderWidth: 1,
    borderRadius: 5,
    backgroundColor: "#f9f9f9",
  },
  predictionTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 10,
    color: "#333",
  },
  predictionItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#eee",
  },
  topPrediction: {
    backgroundColor: "#e6f7ff", // Highlight top prediction with light blue
    borderRadius: 5,
  },
  emptyText: {
    color: "gray",
    textAlign: "center",
    marginVertical: 20,
  },
  predictionLabel: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#555",
  },
  predictionProbability: {
    fontSize: 14,
    color: "gray",
  },
});
