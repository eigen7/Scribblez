// Fetch JSON from the dashboard data API (proxied to the Python server at /api).
export async function getJSON(url: string): Promise<any> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}
