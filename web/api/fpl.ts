const FPL_BASE_URL = "https://fantasy.premierleague.com/api/";

const ALLOWED_PATHS = [
  /^bootstrap-static\/?$/,
  /^fixtures\/?$/,
  /^entry\/\d+\/event\/\d+\/picks\/?$/,
];

type ApiRequest = {
  query: {
    path?: string | string[];
  };
};

type ApiResponse = {
  status: (statusCode: number) => ApiResponse;
  json: (body: unknown) => void;
  send: (body: string) => void;
  setHeader: (name: string, value: string) => void;
};

export default async function handler(req: ApiRequest, res: ApiResponse) {
  const rawPath = Array.isArray(req.query.path) ? req.query.path[0] : req.query.path;
  const path = String(rawPath ?? "").replace(/^\/+/, "");

  if (!ALLOWED_PATHS.some((pattern) => pattern.test(path))) {
    return res.status(400).json({ error: "Unsupported FPL API path" });
  }

  try {
    const upstream = await fetch(`${FPL_BASE_URL}${path}`, {
      headers: {
        "User-Agent": "fpl-intelligence-platform/1.0",
        Accept: "application/json",
      },
    });

    const body = await upstream.text();
    res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate=900");
    res.setHeader("Content-Type", upstream.headers.get("content-type") ?? "application/json");
    return res.status(upstream.status).send(body);
  } catch (error) {
    return res.status(502).json({
      error: error instanceof Error ? error.message : "Failed to fetch FPL API",
    });
  }
}
