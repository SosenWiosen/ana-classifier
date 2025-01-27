import { Button, StyleSheet, Alert, Platform } from "react-native";
import { Text, View } from "@/components/Themed";
import { useSession } from "../../ctx"; // Use the session context
import { router } from "expo-router"; // Import router for navigation

export default function ProfileTab() {
  const { signOut, session } = useSession();

  const handleSignOut = () => {
    if (Platform.OS === "web") {
      // Use `window.confirm` for web dialogs
      const userConfirmed = window.confirm(
        "Are you sure you want to sign out?"
      );
      if (userConfirmed) {
        signOut();
        router.replace("/login");
      }
      return; // Early return for web
    }

    // Use native Alert for mobile platforms
    Alert.alert("Sign Out", "Are you sure you want to sign out?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Sign Out",
        style: "destructive",
        onPress: () => {
          signOut();
          router.replace("/login");
        },
      },
    ]);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Profile</Text>
      <Text style={styles.welcomeText}>
        Welcome, {session?.username || "Guest"}
      </Text>
      <View
        style={styles.separator}
        lightColor="#eee"
        darkColor="rgba(255,255,255,0.1)"
      />
      <Button title="Sign Out" onPress={handleSignOut} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 10,
  },
  welcomeText: {
    fontSize: 16,
    marginBottom: 20,
  },
  separator: {
    marginVertical: 30,
    height: 1,
    width: "80%",
  },
});
