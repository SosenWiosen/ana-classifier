import React from "react";
import { FlatList, StyleSheet, Text, View } from "react-native";
import HistoryItem from "./HistoryItem";

interface HistoryData {
  image_name: string;
  prediction_result: string; // JSON string containing predictions
  timestamp: string; // ISO string
}

interface HistoryListProps {
  history: HistoryData[]; // Array of history items
}

const HistoryList: React.FC<HistoryListProps> = ({ history }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.historyTitle}>History</Text>
      <FlatList
        contentContainerStyle={styles.historyList}
        data={history}
        keyExtractor={(item, index) => index.toString()} // Use index as fallback key
        renderItem={({ item }) => (
          <HistoryItem
            imageName={item.image_name}
            predictionResult={item.prediction_result}
            timestamp={item.timestamp}
          />
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  historyTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginTop: 30,
    marginBottom: 10,
    marginLeft: 10, // Add some padding for the title
  },
  historyList: {
    paddingBottom: 50, // Add some space at the bottom of the list
  },
});

export default HistoryList;
