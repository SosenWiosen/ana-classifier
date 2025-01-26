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

export const SessionProvider: React.FC<React.PropsWithChildren> = ({
  children,
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [token, setToken] = useStorageState("token");
  const [username, setUsername] = useStorageState("username");

  // Sign In
  const signIn = async (username: string, password: string) => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${API.URL}/login`, {
        username,
        password,
      });

      const { token, exp } = response.data; // Save token and expiry date if available
      setToken(token);
      setUsername(username);
      if (exp) {
        await setStorageItemAsync("token_exp", exp); // Store `exp` for later use
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        console.error(
          "Sign-in error:",
          error.response?.data?.message || error.message
        );
        throw error;
      } else {
        console.error("Sign-in error:", (error as Error).message);
        throw error;
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Sign Out
  const signOut = async () => {
    await setStorageItemAsync("token", null);
    await setStorageItemAsync("username", null);
    await setStorageItemAsync("token_exp", null); // Clear the token expiry
    setUsername(null);
    setToken(null);
  };

  // Monitor Token Expiry
  useEffect(() => {
    const checkTokenExpiry = async () => {
      const expTimestamp = await localStorage.getItem("token_exp");
      if (!expTimestamp) return;

      const expiryDate = new Date(parseInt(expTimestamp, 10) * 1000); // Convert `exp` to a Date object
      const now = new Date();

      if (expiryDate <= now) {
        console.warn("Your session has expired. Signing out...");
        signOut();
      } else {
        const timeout = expiryDate.getTime() - now.getTime();
        setTimeout(() => {
          console.warn("Your session has expired. Signing out...");
          signOut();
        }, timeout);
      }
    };

    checkTokenExpiry();
  }, [token[1]]);

  return (
    <AuthContext.Provider
      value={{
        signIn,
        signOut,
        session: { token: token[1], username: username[1] },
        isLoading,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
