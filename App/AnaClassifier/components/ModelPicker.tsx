import React from "react";
import { StyleSheet } from "react-native";
import { Picker } from "@react-native-picker/picker";
import { View } from "./Themed";
import { Text } from "react-native";

type ModelPickerProps = {
  models: { name: string }[]; // An array of available models
  selectedModel: string; // The currently selected model
  onSelectModel: (model: string) => void; // Callback to handle model selection
};

const ModelPicker: React.FC<ModelPickerProps> = ({
  models,
  selectedModel,
  onSelectModel,
}) => {
  const hasModels = models.length > 0; // Check if models are loaded

  return (
    <View>
      <Text>Choose the model you want to use for the prediction:</Text>
      {!hasModels ? ( // If no models are available, display a message
        <Text style={styles.noModelsMessage}>No models available</Text>
      ) : (
        <Picker
          selectedValue={selectedModel}
          onValueChange={onSelectModel}
          style={styles.picker}
          itemStyle={{ color: "black" }}
          enabled={hasModels} // Disable the Picker if no models are available
        >
          {models.map((model, index) => (
            <Picker.Item key={index} label={model.name} value={model.name} />
          ))}
        </Picker>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  picker: {
    color: "black", // This ensures text is visible on Android
    backgroundColor: "white", // Helps avoid transparency issues on Android
  },
  noModelsMessage: {
    marginTop: 10,
    fontSize: 16,
    fontStyle: "italic",
    color: "red", // Highlight the message with a distinct color
  },
});

export default ModelPicker;
