/* Fieldnote service worker — caches ONLY the static app shell so the app opens
   offline. It deliberately never touches /api/* so identifications and health
   are always live and never served stale. */
var CACHE = "fieldnote-shell-v1";
var SHELL = [
  "/",
  "/static/index.html",
  "/static/style.css",
  "/static/app.js",
  "/static/manifest.webmanifest",
  "/static/icon.svg"
];

self.addEventListener("install", function (e) {
  e.waitUntil(caches.open(CACHE).then(function (c) { return c.addAll(SHELL); }).then(function () { return self.skipWaiting(); }));
});

self.addEventListener("activate", function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (k) { if (k !== CACHE) return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener("fetch", function (e) {
  var req = e.request;
  if (req.method !== "GET") return;
  var url = new URL(req.url);

  // Never cache the API — always go to the network for live results.
  if (url.pathname.indexOf("/api/") === 0) return;

  // Same-origin static shell: cache-first, fall back to network, then update.
  if (url.origin === self.location.origin) {
    e.respondWith(
      caches.match(req).then(function (hit) {
        if (hit) return hit;
        return fetch(req).then(function (res) {
          if (res && res.ok && res.type === "basic") {
            var copy = res.clone();
            caches.open(CACHE).then(function (c) { c.put(req, copy); });
          }
          return res;
        }).catch(function () {
          // offline and uncached: fall back to the app shell for navigations
          if (req.mode === "navigate") return caches.match("/static/index.html");
        });
      })
    );
  }
});
