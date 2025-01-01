import React, { useState } from "react";
import axios from "axios";


const AuthContext = React.createContext<{
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => void;
  session?: { token: string | null; username: string | null };
  isLoading: boolean;
}>({
  signIn: async () => {},
  signOut: () => {},
  session: { token: null, username: null },
  isLoading: false,
});

export function useSession() {
  const value = React.useContext(AuthContext);
  if (!value) {
    throw new Error("useSession must be wrapped in a <SessionProvider />");
  }
  return value;
}

export const SessionProvider: React.FC<React.PropsWithChildren> = (props) => {
  const [isLoading, setIsLoading] = useState(false);
  const [session, setSession] = useState<{ token: string | null; username: string | null }>({
    token: null,
    username: null,
  });

  const signIn = async (username: string, password: string) => {
    setIsLoading(true);
    try {
      const response = await axios.post("http://localhost:5000/login", {
        username,
        password,
      });

      // The backend returns a JWT token
      const { token } = response.data;

      // Save the token and username into the React state (and optionally localStorage/sessionStorage)
      setSession({ token, username });

      // Optionally, persist session in localStorage or cookies
      localStorage.setItem("token", token);
      localStorage.setItem("username", username);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        console.error("Sign-in error:", error.response?.data?.message || error.message);
        throw error;
      } else {
        console.error("Sign-in error:", (error as Error).message);
        throw error;
      }
    } finally {
      setIsLoading(false);
    }
  };

  const signOut = () => {
    console.log('signing out')
    // Clear session state
    setSession({ token: null, username: null });

    // Remove session data from localStorage or cookies
    localStorage.removeItem("token");
    localStorage.removeItem("username");
  };

  // Check if a session exists on mount (e.g., JWT in localStorage)
  React.useEffect(() => {
    const token = localStorage.getItem("token");
    const username = localStorage.getItem("username");
    if (token && username) {
      setSession({ token, username });
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{
        signIn,
        signOut,
        session,
        isLoading,
      }}
    >
      {props.children}
    </AuthContext.Provider>
  );
};