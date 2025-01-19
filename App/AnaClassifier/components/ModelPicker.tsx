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
  return (
    <View>
      <Text> Choose the model you want to use for the prediction: </Text>
      <Picker
        selectedValue={selectedModel}
        onValueChange={onSelectModel}
        style={styles.picker}
      >
        {models.map((model, index) => (
          <Picker.Item key={index} label={model.name} value={model.name} />
        ))}
      </Picker>
    </View>
  );
};

const styles = StyleSheet.create({
  picker: {
    height: 50,
    width: "100%",
  },
});

export default ModelPicker;
