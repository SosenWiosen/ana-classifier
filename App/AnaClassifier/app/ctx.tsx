import React, { useState, useContext, useEffect } from "react";
import axios from "axios";
import { API } from "@/constants/API";
import { useStorageState, setStorageItemAsync } from "./useStorageState";

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
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error("useSession must be wrapped in a <SessionProvider />");
  }
  return value;
}

export const SessionProvider: React.FC<React.PropsWithChildren> = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [token, setToken] = useStorageState('token');
  const [username, setUsername] = useStorageState('username');

  const signIn = async (username: string, password: string) => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${API.URL}/login`, { username, password });
      const { token } = response.data;
      setToken(token);
      setUsername(username);
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

  const signOut = async () => {
    await setStorageItemAsync('token', null);
    await setStorageItemAsync('username', null);
    setUsername(null);
    setToken(null);
  };

  // Check if a session exists on mount
  useEffect(() => {
    if (token[1] && username[1]) {
      // You may want to validate or refresh the token here if necessary
    }
  }, [token[1], username[1]]);

  return (
    <AuthContext.Provider
      value={{
        signIn,
        signOut,
        session: { token: token[1], username: username[1] },
        isLoading
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};