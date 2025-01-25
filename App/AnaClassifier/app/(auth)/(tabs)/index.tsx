import React, { useEffect, useState } from "react";
import {
  StyleSheet,
  FlatList,
  Button,
  Image,
  Alert,
  Dimensions,
  Platform,
  ScrollView,
} from "react-native";
import { View, Text } from "@/components/Themed";
import * as ImagePicker from "expo-image-picker";
import axios from "axios";
import { useSession } from "@/app/ctx";
import { API } from "@/constants/API";
import ModelPicker from "@/components/ModelPicker";
import {
  PredictionItem,
  PredictionResults,
} from "@/components/PredictionResults";

type HistoryItem = {
  image_name: string;
  timestamp: string;
  prediction_result: string;
};
type ModelItem = {
  name: string;
};

// Get the width of the current window
const DEVICE_WIDTH = Dimensions.get("window").width;

// Conditionally calculate the width for desktop or browser environments
const isWebDesktop = Platform.OS === "web" && DEVICE_WIDTH > 768; // Adjust the threshold as needed

export default function HomeScreen() {
  const { session } = useSession();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selectedImage, setSelectedImage] =
    useState<ImagePicker.ImagePickerAsset | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isUploading, setIsUploading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionItem[] | null>(null);

  // Fetch available models from the backend
  const fetchModels = async () => {
    try {
      const response = await axios.get(API.URL + "/models", {
        headers: { "x-access-token": session?.token },
      });
      setModels(
        response.data.available_models.map((modelName: any) => ({
          name: modelName,
        }))
      );
      if (response.data.available_models.length > 0) {
        setSelectedModel(response.data.available_models[0]);
      }
    } catch (error: any) {
      console.error("Error fetching models:", error.message);
      Alert.alert("Error", "Unable to fetch models.");
    }
  };

  useEffect(() => {
    fetchHistory();
    fetchModels();
  }, []);

  // Fetch prediction history from the backend
  const fetchHistory = async () => {
    try {
      const response = await axios.get(API.URL + "/history", {
        headers: { "x-access-token": session?.token },
      });
      setHistory(response.data.history); // Update state with the fetched history
    } catch (error: any) {
      console.error("Error fetching history:", error.message);
      Alert.alert("Error", "Unable to fetch prediction history.");
    }
  };

  // Pick an image using the image picker
  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });
    if (!result.canceled) {
      setSelectedImage(result.assets[0]); // Store the selected image
    }
  };

  // Upload the picked image to the backend
  const uploadImage = async () => {
    if (!selectedImage) {
      Alert.alert("Error", "Please select an image first.");
      return;
    }
    if (!selectedModel) {
      Alert.alert("Error", "Please select a model first.");
      return;
    }
    setIsUploading(true);
    const convertToBase64 = async () => {
      const response = await fetch(selectedImage.uri);
      const blob = await response.blob();
      return new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          resolve(reader.result as string);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    };
    try {
      const base64 = await convertToBase64();
      const base64Data = base64.replace(/^data:image\/\w+;base64,/, "");
      const payload = {
        image: base64Data,
        image_filename: selectedImage.fileName,
        model_name: selectedModel,
      };
      const response = await axios.post(API.URL + "/predict", payload, {
        headers: {
          "x-access-token": session?.token,
          "Content-Type": "application/json",
        },
      });
      console.log("Prediction Response:", response.data);
      setPrediction(JSON.parse(response.data.predictions));
      fetchHistory();
    } catch (error: any) {
      console.error("Error uploading image:", error.message);
      Alert.alert(
        "Error",
        error.response?.data?.message || "Error uploading image."
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.scrollContainer}>
      <View style={styles.container}>
        {/* Section: Title */}
        <Text style={styles.title}>Predict the ANA Pattern of your image</Text>
        <Text style={styles.instructions}>
          1. Select the model from the dropdown.
        </Text>
        <Text style={styles.instructions}>
          2. Pick the image by pressing the button.
        </Text>
        <Text style={styles.instructions}>
          3. Upload the image to the server.
        </Text>
        <Text style={styles.instructions}>
          4. Wait for the results which will be displayed below in the History
          field.
        </Text>

        {/* Section: Image Picker & Upload */}
        <View style={styles.buttonsContainer}>
          <Button title="Pick an Image" onPress={pickImage} />
          <View style={styles.imagePlaceholder}>
            {selectedImage ? (
              <Image
                source={{ uri: selectedImage.uri }}
                style={styles.selectedImage}
              />
            ) : (
              <Text style={styles.placeholderText}>No Image Selected</Text>
            )}
          </View>

          {/* Model Selector */}
          <ModelPicker
            models={models}
            selectedModel={selectedModel}
            onSelectModel={(model) => setSelectedModel(model)}
          />
          <View style={styles.uploadButton}>
            <Button
              title={isUploading ? "Uploading..." : "Upload & Predict"}
              onPress={uploadImage}
              disabled={isUploading}
            />
          </View>
        </View>

        {/* Horizontal Divider */}
        <View style={styles.divider} />
        {/* Section: Show Prediction Result */}
        {prediction && <PredictionResults predictions={prediction} />}

        {/* Section: Prediction History */}
        <View
          style={{
            width: isWebDesktop ? "80%" : "100%",
            alignSelf: "center",
          }}
        >
          <Text style={styles.historyTitle}>History</Text>
          <FlatList
            contentContainerStyle={styles.historyList}
            data={history}
            keyExtractor={(item, index) => index.toString()}
            renderItem={({ item }) => (
              <View style={styles.historyItem}>
                <Text style={styles.historyText}>Image: {item.image_name}</Text>
                <Text style={styles.historyText}>
                  Prediction: {item.prediction_result}
                </Text>
                <Text style={styles.historyTimestamp}>
                  Date: {new Date(item.timestamp).toLocaleString()}
                </Text>
              </View>
            )}
          />
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scrollContainer: {
    flexGrow: 1, // Ensures the content is scrollable even if it doesn't fill the screen
  },
  container: {
    flex: 1,
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
    marginBottom: 20,
    textAlign: "center",
  },
  instructions: {
    fontSize: 16,
    marginBottom: 5,
    color: "gray",
    textAlign: "center",
  },
  buttonsContainer: {
    marginBottom: 20,
    alignItems: "center",
  },
  uploadButton: {
    marginTop: 10, // Add spacing between the buttons
  },
  imagePlaceholder: {
    width: isWebDesktop ? "50%" : "100%", // Conditionally provide width for desktop vs mobile
    height: 300, // Fixed height for the image placeholder
    backgroundColor: "#f0f0f0", // Light gray background for the placeholder
    justifyContent: "center", // Center content vertically
    alignItems: "center", // Center content horizontally
    marginVertical: 20, // Provide spacing above and below
    borderWidth: 1,
    borderColor: "#ccc", // Placeholder border color
    borderRadius: 5, // Optional: Make corners rounded
    alignSelf: "center", // Ensure it is centered horizontally
  },
  selectedImage: {
    flex: 1, // Ensures the image fills the available space
    width: "100%", // Takes the full width of the placeholder
    resizeMode: "contain", // Fits the image into the frame without cropping
  },
  placeholderText: {
    fontSize: 16,
    color: "#aaa", // Light gray text color to match the placeholder style
  },
  picker: {
    height: 50,
    width: "100%",
  },
  prediction: {
    fontSize: 18,
    fontWeight: "bold",
    marginVertical: 20,
    color: "green",
    textAlign: "center",
  },
  divider: {
    borderBottomWidth: 1,
    borderColor: "#cccccc", // Light gray line
    marginVertical: 20, // Spacing above and below the divider
    width: "100%",
  },
  historyTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginTop: 30,
    marginBottom: 10,
  },
  historyList: {
    paddingBottom: 50, // Add some space at the bottom of the list
  },
  historyItem: {
    borderBottomWidth: 1,
    borderColor: "#ccc",
    paddingVertical: 10,
  },
  historyText: {
    fontSize: 16,
  },
  historyTimestamp: {
    fontSize: 12,
    color: "gray",
  },
});
