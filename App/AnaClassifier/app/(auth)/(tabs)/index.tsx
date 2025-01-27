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
  ActivityIndicator, // Import ActivityIndicator
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
import HistoryList from "@/components/HistoryList";

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
  const { session } = useSession(); // Get token from context
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selectedImage, setSelectedImage] =
    useState<ImagePicker.ImagePickerAsset | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isUploading, setIsUploading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionItem[] | null>(null);
  const [fetchingPrediction, setFetchingPrediction] = useState(false); // NEW: State for fetching
  const [errorMessage, setErrorMessage] = useState<string | null>(null); // NEW: State for error message

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
      setPrediction(null); // Reset previous predictions
      setErrorMessage(null); // Clear error message
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
    setIsUploading(true); // Block the upload button
    setFetchingPrediction(true); // Start fetching state
    setErrorMessage(null); // Clear any previous error message
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
      setPrediction(response.data.prediction); // Show predictions
      fetchHistory();
    } catch (error: any) {
      console.error("Error uploading image:", error.message);
      setErrorMessage(
        // Display error message
        error.response?.data?.message || "Failed to make a prediction."
      );
    } finally {
      setIsUploading(false); // Enable buttons again
      setFetchingPrediction(false); // Stop fetching state
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.scrollContainer}>
      <View style={styles.container}>
        {/* Title */}
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

        {/* Image Picker and Upload */}
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

        {/* Loading Indicator */}
        {fetchingPrediction && (
          <ActivityIndicator
            size="large"
            color="#0000ff"
            style={styles.loader}
          />
        )}

        {/* Error Message */}
        {errorMessage && <Text style={styles.errorText}>{errorMessage}</Text>}

        {/* Show Predictions */}
        {prediction && <PredictionResults predictions={prediction} />}

        {/* Prediction History */}
        <View
          style={{
            width: isWebDesktop ? "80%" : "100%",
            alignSelf: "center",
          }}
        >
          <HistoryList history={history} />
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scrollContainer: {
    flexGrow: 1,
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
    marginTop: 10,
  },
  imagePlaceholder: {
    width: isWebDesktop ? "50%" : "100%",
    height: 300,
    backgroundColor: "#f0f0f0",
    justifyContent: "center",
    alignItems: "center",
    marginVertical: 20,
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 5,
    alignSelf: "center",
  },
  selectedImage: {
    flex: 1,
    width: "100%",
    resizeMode: "contain",
  },
  placeholderText: {
    fontSize: 16,
    color: "#aaa",
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
    borderColor: "#cccccc",
    marginVertical: 20,
    width: "100%",
  },
  loader: {
    marginVertical: 10, // Space around the loader
  },
  errorText: {
    color: "red",
    textAlign: "center",
    marginVertical: 10, // Space around the error text
  },
});
