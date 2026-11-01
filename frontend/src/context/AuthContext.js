import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import apiClient, { getStoredToken, setStoredToken } from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(getStoredToken());
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const logout = useCallback(() => {
    setStoredToken(null);
    setToken(null);
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    const stored = getStoredToken();
    if (!stored) {
      setUser(null);
      setLoading(false);
      return null;
    }
    try {
      const response = await apiClient.get('/auth/me');
      setUser(response.data);
      setToken(stored);
      return response.data;
    } catch {
      logout();
      return null;
    } finally {
      setLoading(false);
    }
  }, [logout]);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const login = useCallback(async (email, password) => {
    const response = await apiClient.post('/auth/login', { email, password });
    const nextToken = response.data.token;
    setStoredToken(nextToken);
    setToken(nextToken);
    setUser({
      id: response.data.id,
      name: response.data.name,
      email: response.data.email,
      role: response.data.role,
    });
    return response.data;
  }, []);

  const register = useCallback(async (name, email, password) => {
    const response = await apiClient.post('/auth/register', {
      name,
      email,
      password,
      role: 'content_writer',
    });
    const nextToken = response.data.token;
    setStoredToken(nextToken);
    setToken(nextToken);
    setUser({
      id: response.data.id,
      name: response.data.name,
      email: response.data.email,
      role: response.data.role,
    });
    return response.data;
  }, []);

  const value = useMemo(
    () => ({
      token,
      user,
      loading,
      isAuthenticated: Boolean(token && user),
      login,
      register,
      logout,
      refreshUser,
    }),
    [token, user, loading, login, register, logout, refreshUser]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
