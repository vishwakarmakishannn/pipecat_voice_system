const developmentApiBase = `${window.location.protocol}//${window.location.hostname}:7860`;

// Production traffic stays on the page origin and is forwarded by Nginx. This
// reuses the browser's existing TLS connection and avoids CORS preflights. The
// explicit override remains useful for split-origin or hosted deployments.
export const API_BASE = (
  import.meta.env.VITE_API_BASE || (import.meta.env.DEV ? developmentApiBase : window.location.origin)
).replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, status = 0, data = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

export async function apiRequest(endpoint, options = {}) {
  const { auth = true, timeoutMs = 10000, retries = 0, ...fetchOptions } = options;
  const token = auth ? localStorage.getItem('aura_token') : null;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  
  const headers = {
    ...fetchOptions.headers,
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      ...fetchOptions,
      headers,
      signal: fetchOptions.signal || controller.signal,
    });
  
    if (res.status === 401 && auth) {
      localStorage.removeItem('aura_token');
      window.dispatchEvent(new Event('logout'));
      throw new ApiError('Your session has expired. Please sign in again.', 401);
    }

    if (!res.ok) {
      const data = await res.clone().json().catch(() => null);
      throw new ApiError(data?.detail || `Request failed (${res.status})`, res.status, data);
    }
  
    return res;
  } catch (error) {
    const retryable = error.name === 'AbortError' || error instanceof TypeError || error.status >= 500;
    if (retries > 0 && retryable) {
      return apiRequest(endpoint, { ...options, retries: retries - 1 });
    }
    if (error.name === 'AbortError') throw new ApiError('Request timed out');
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export function fetchWithAuth(endpoint, options = {}) {
  const method = (options.method || 'GET').toUpperCase();
  return apiRequest(endpoint, { retries: method === 'GET' ? 1 : 0, ...options });
}
