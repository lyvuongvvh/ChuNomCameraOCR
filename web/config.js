// Where app.js sends its POST /v1/ocr requests.
//
// Leave as "" (default) when this page is served by the SAME FastAPI process as the API
// (api/main.py's StaticFiles mount) - relative paths, same-origin, no CORS needed.
//
// Set to the API's full origin (e.g. "https://nomocr-api.example.com") when this page is hosted
// separately from the API (e.g. web/ on Cloudflare Pages, the API on a Cloudflare Tunnel URL
// pointed at a Docker host like Unraid) - the API then needs ALLOWED_ORIGINS set to this page's
// own origin (see api/config.py) for the browser's CORS check to pass, and the API must be
// reachable over HTTPS, not just plain HTTP, or an HTTPS-served page like a Cloudflare Pages site
// will have this fetch blocked as mixed content.
const API_BASE_URL = "";
