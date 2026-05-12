export async function fetchFplApi<T>(path: string): Promise<T> {
  const normalizedPath = path.replace(/^\/+/, "");
  const response = await fetch(`/api/fpl?path=${encodeURIComponent(normalizedPath)}`);

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `FPL API request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}
