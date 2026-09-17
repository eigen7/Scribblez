// A failed call's error carries the server's reason: the API answers an
// expected failure (a refused launch, a bad parameter) with 400 and
// {"error": <sentence>}, which is what the operator needs to see.
async function failure(r: Response, url: string): Promise<Error> {
  let reason = '';
  try {
    reason = (await r.json()).error ?? '';
  } catch {
    // not JSON (a proxy error page): the status is all there is
  }
  return new Error(reason || `${r.status} ${url}`);
}

// Fetch JSON from the dashboard data API (proxied to the Python server at /api).
export async function getJSON(url: string): Promise<any> {
  const r = await fetch(url);
  if (!r.ok) throw await failure(r, url);
  return r.json();
}

// POST a JSON body to the data API (used by the Controls tab to set live knobs).
export async function postJSON(url: string, body: unknown): Promise<any> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw await failure(r, url);
  return r.json();
}
