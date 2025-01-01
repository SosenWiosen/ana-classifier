import React, { useEffect, useState } from "react";
import { StyleSheet, FlatList, Button, Image, Alert } from "react-native";
import { View, Text } from "@/components/Themed";
import * as ImagePicker from "expo-image-picker";
import axios from "axios";
import { useSession } from "@/app/ctx";
import { API } from "@/constants/API";
type HistoryItem = {
  image_name: string;
  timestamp: string;
  prediction_result: string;
};

export default function HomeScreen() {
  const { session } = useSession(); // Access the user's session (JWT token)
  const [history, setHistory] = useState<HistoryItem[]>([]); // State to store prediction history
  const [selectedImage, setSelectedImage] = useState<ImagePicker.ImagePickerAsset | null>(null); // State to store the picked image
  const [isUploading, setIsUploading] = useState(false); // State to manage upload state
  const [prediction, setPrediction] = useState<string | null>(null); // Store prediction result

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
  
    setIsUploading(true); // Show uploading state
  
    // Convert image to base64
    const convertToBase64 = async () => {
      const response = await fetch(selectedImage.uri);
      const blob = await response.blob();
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          resolve(reader.result);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    };
  
    try {
      const base64 = await convertToBase64();
      const base64Data = (base64 as string).replace(/^data:image\/\w+;base64,/, ""); // Strip out the metadata
  
      // Prepare the JSON payload
      const payload = {
        image: base64Data,
      };
  
      // Send the POST request to the backend
      const response = await axios.post(API.URL + "/predict", payload, {
        headers: {
          "x-access-token": session?.token, // JWT token for authentication
          "Content-Type": "application/json", // Set the content type to application/json
        },
      });
  
      setPrediction(response.data.prediction); // Handle prediction response
      fetchHistory(); // Fetch history after prediction
    } catch (error: any) {
      // Display error message from server
      Alert.alert("Error", error.response?.data?.message || "Error uploading image.");
    } finally {
      setIsUploading(false); // Reset uploading state
    }
  };

  // Fetch history when the component loads
  useEffect(() => {
    fetchHistory();
  }, []);

  return (
    <View style={styles.container}>
      {/* Section: Title */}
      <Text style={styles.title}>Predict the ANA Pattern of your image.</Text>
      <Text style={styles.instructions}> 1. Pick the image by pressing the button.</Text>
      <Text style={styles.instructions}> 2. Upload image to the server.</Text>
      <Text style={styles.instructions}> 3. Wait for the results which will be displayer below in the History field.</Text>
      {/* Section: Image Picker & Upload */}
      <View style={styles.buttonsContainer}>
        <Button title="Pick an Image" onPress={pickImage} />
        {selectedImage && (
          <Image
            source={{ uri: selectedImage.uri }}
            style={styles.selectedImage}
          />
        )}
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
      {prediction && (
        <Text style={styles.prediction}>
          Prediction: {prediction}
        </Text>
      )}

      {/* Section: Prediction History */}
      <Text style={styles.historyTitle}>History</Text>
      <FlatList
        contentContainerStyle={styles.historyList}
        data={history}
        keyExtractor={(item, index) => index.toString()}
        renderItem={({ item }) => (
          <View style={styles.historyItem}>
            <Text style={styles.historyText}>
              Image: {item.image_name}
            </Text>
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
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
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
  selectedImage: {
    width: "100%",
    height: 200,
    resizeMode: "contain",
    marginVertical: 20,
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