/* Fieldnote service worker — caches ONLY the static app shell so the app opens
   on a bad signal. It deliberately never touches /api/* so identifications and
   health are always live and never served stale.

   Strategy: NETWORK-FIRST for the shell (was cache-first, which meant a phone
   that had once loaded the app could never receive a frontend update — a
   deployed app needs fixes to actually reach users). Falling back to cache
   keeps it usable in the field:

     online            -> always the freshest code
     slow signal       -> cached shell after SLOW_MS, network still refreshes it
     offline / failure -> cached shell (identification itself needs the network)

   Bump VERSION on any shell change; `activate` purges every other cache. */
var VERSION = "v2";
var CACHE = "fieldnote-shell-" + VERSION;
var SLOW_MS = 3000;   // fall back to cache if the network is this slow
var SHELL = [
  "/",
  "/static/index.html",
  "/static/style.css",
  "/static/app.js",
  "/static/manifest.webmanifest",
  "/static/icon.svg"
];

self.addEventListener("install", function (e) {
  e.waitUntil(
    caches.open(CACHE)
      .then(function (c) { return c.addAll(SHELL); })
      .then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener("activate", function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (k) {
        if (k !== CACHE) return caches.delete(k);
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

function shellResponse(req) {
  // Bypass the browser's HTTP cache for assets. Without this, "network-first"
  // still lands in the HTTP cache — and since the server sends no Cache-Control,
  // the browser applies HEURISTIC freshness and can serve a stale app.js without
  // ever asking the server. "no-cache" forces revalidation (a cheap 304 via
  // ETag when unchanged). Navigation requests can't be re-constructed this way,
  // so those rely on the Cache-Control header Caddy adds.
  var netReq = req;
  if (req.mode !== "navigate") {
    try { netReq = new Request(req.url, { cache: "no-cache", credentials: "same-origin" }); }
    catch (err) { netReq = req; }
  }

  // Network wins when it answers in time, and always refreshes the cache.
  var fromNetwork = fetch(netReq).then(function (res) {
    if (res && res.ok && res.type === "basic") {
      var copy = res.clone();
      caches.open(CACHE).then(function (c) { c.put(req, copy); });
    }
    return res;
  });

  // Resolves with the cached copy only if the network is slow AND we have one.
  // If there's no cached copy this stays pending on purpose, letting the
  // network settle the race either way.
  var slowFallback = new Promise(function (resolve) {
    setTimeout(function () {
      caches.match(req).then(function (hit) { if (hit) resolve(hit); });
    }, SLOW_MS);
  });

  return Promise.race([fromNetwork, slowFallback]).catch(function () {
    // Network failed outright: serve cache, or the shell for a navigation.
    return caches.match(req).then(function (hit) {
      if (hit) return hit;
      if (req.mode === "navigate") return caches.match("/static/index.html");
      return Response.error();
    });
  });
}

self.addEventListener("fetch", function (e) {
  var req = e.request;
  if (req.method !== "GET") return;
  var url = new URL(req.url);

  // Never cache the API — always go to the network for live results.
  if (url.pathname.indexOf("/api/") === 0) return;

  if (url.origin === self.location.origin) {
    e.respondWith(shellResponse(req));
  }
});
