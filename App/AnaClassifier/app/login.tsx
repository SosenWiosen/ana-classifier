import { useState } from "react";
import { Button, StyleSheet, TextInput, Alert } from "react-native";
import { Text, View } from "@/components/Themed";
import { useSession } from "./ctx"; // Updated provider
import { router } from "expo-router";

export default function Login() {
  const { signIn, isLoading } = useSession(); // Destructure `signIn` and `isLoading`
  
  const [username, setUsername] = useState(""); // State for username
  const [password, setPassword] = useState(""); // State for password
  const [errorMessage, setErrorMessage] = useState<string | null>(null); // State for error message (new)

  const handleLogin = async () => {
    // Clear existing error messages before attempting to log in
    setErrorMessage(null);

    if (!username || !password) {
      setErrorMessage("Please enter both username and password."); // Validation error
      return;
    }

    try {
      // Attempt to sign in using the useSession `signIn` method
      await signIn(username, password);

      // If successful, navigate to home (or another protected screen)
      router.replace("/");
    } catch (error: any) {
      // Handle errors (login failure) and show helpful hints to the user
      console.error("Login Error:", error.message);

      // Set a user-friendly error message
      setErrorMessage(
        error.response?.data?.message || "Invalid username or password. Please try again."
      );

      // Optionally, display an alert for more immediate attention
      Alert.alert(
        "Login Failed",
        error.response?.data?.message || "An error occurred while logging in."
      );
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Ana Classifier</Text>
      <Text style={styles.paragraph}>
        This is an app to test classification of ANA patterns.
      </Text>
      <View
        style={styles.separator}
        lightColor="#eee"
        darkColor="rgba(255,255,255,0.1)"
      />

      {/* Username Input */}
      <TextInput
        placeholder="Username"
        style={styles.input}
        value={username}
        onChangeText={setUsername} // Update state on text input change
        editable={!isLoading} // Disable input when loading
      />

      {/* Password Input */}
      <TextInput
        placeholder="Password"
        secureTextEntry
        style={styles.input}
        value={password}
        onChangeText={setPassword} // Update state on password input change
        editable={!isLoading} // Disable input when loading
      />

      {/* Error Message Section */}
      {errorMessage && <Text style={styles.error}>{errorMessage}</Text>}

      {/* Login Button */}
      <Button
        title={isLoading ? "Logging in..." : "Login"} // Show loading indicator text
        onPress={handleLogin} // Call handleLogin on press
        disabled={isLoading} // Disable button during loading
      />
    </View>
  );
}

// Styles applied to different components in the UI
const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
  },
  paragraph: {
    margin: 24,
    fontSize: 18,
    textAlign: "center",
  },
  separator: {
    marginVertical: 30,
    height: 1,
    width: "80%",
  },
  input: {
    width: "80%",
    borderWidth: 1,
    borderColor: "#000",
    padding: 10,
    margin: 10,
    borderRadius: 4,
  },
  // Style applied to the error message
  error: {
    color: "red",
    marginVertical: 10,
    textAlign: "center",
  },
});