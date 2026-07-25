# Deployment plan — Fieldnote on a real URL

Goal: the app reachable from a phone **anywhere**, not just on home wifi with a
Mac awake. That's the point — real field testing (a yard, a trail, a nursery) is
where the next round of failures comes from, and it can't happen on `localhost`.

Written 2026-07-25, after Workstream B shipped the 2,510-class model.

---

## 0. Measurements that decide the plan

Taken from the running local server (not guesses):

| metric | measured | implication |
|--------|----------|-------------|
| server process RSS | **394 MB** | 1 GB droplet is too tight; 2 GB is comfortable |
| identify latency | **~0.10 s** (1 image, CPU) | fast enough as-is |
| model.pt | 90 MB | + 19 MB OOD bank, 169 B temperature |
| torch (CPU-only wheel) | ~200 MB installed | vs ~800 MB+ with bundled CUDA — use the CPU index |

**Conclusion: no ONNX conversion needed.** It was on the table to shrink memory
and latency, but at 394 MB / 0.10 s neither is a problem. Skip the complexity;
revisit only if we move to a $6 box or the model grows a lot.

---

## 1. Decisions to make before starting

1. **Domain — required, not optional.** The PWA's camera capture, service
   worker, and add-to-home-screen **only work over HTTPS**, and HTTPS needs a
   domain name (a bare IP can't get a trusted certificate). Options:
   - *Buy one* (~$12/yr at Cloudflare/Namecheap) — e.g. `fieldnote.app`. Cleanest,
     looks real on a résumé/portfolio.
   - *Free subdomain* (DuckDNS, e.g. `fieldnote.duckdns.org`) — works fine with
     Let's Encrypt, zero cost, slightly less polished.
2. **Public or private?** A public URL means anyone can hit the model and burn
   your CPU. For a portfolio app that's usually fine, but consider a simple rate
   limit, or keep the URL unadvertised.
3. **Provider.** DigitalOcean is the path of least resistance (already have an
   account, the volume, and the CLI muscle memory). ~$12/mo for 2 GB/1 vCPU/50 GB.

---

## 1b. Co-hosting the NFL project on the same box

Both portfolio apps should share **one droplet and one domain**. Subdomains are
free, Caddy routes by hostname, and the two workloads barely overlap in time.

**Measured/estimated footprints:**

| app | runtime memory | serving data | notes |
|-----|----------------|--------------|-------|
| Fieldnote (plant ID) | **394 MB** (measured) | 109 MB model artifacts | torch + one model in memory |
| NFL value analysis | ~350–500 MB (est.) | ~50 MB — `data/processed/` + `outputs/` + 3 raw CSVs | Streamlit + pandas + plotly |

The NFL repo is 2.3 GB locally, but that's notebooks, the 425 MB `data/raw/`
pull, and a venv — **none of which the served app needs.** Ship only
`app/`, `src/`, `data/processed/`, `outputs/`, and the three raw CSVs the app
actually reads.

**Sizing:** together ~900 MB + ~400 MB OS ≈ **1.3 GB**. Start on the **2 GB
droplet ($12/mo)**; DigitalOcean can resize RAM/CPU up later without rebuilding
(disk growth is one-way, RAM/CPU is reversible). Two separate 2 GB droplets
would be $24/mo — so co-hosting **saves ~$144/yr** and halves the maintenance.

```
                    ┌─ fieldnote.<domain>  → :8700  uvicorn  (plant ID)
phone/laptop ─TLS─ Caddy
                    └─ nfl.<domain>        → :8501  streamlit (NFL value)
```

**Streamlit-specific gotcha:** Streamlit talks over **websockets**, so the proxy
must forward upgrade requests. Caddy's `reverse_proxy` does this automatically —
one of the reasons to prefer it over hand-rolled nginx config here. Run it with
`--server.headless true --server.address 127.0.0.1` and set
`--server.baseUrlPath` only if serving under a path instead of a subdomain.

**On "more polished than Streamlit":** that's a *frontend rebuild*, not a hosting
problem — and this repo is the template. Fieldnote's stack (FastAPI JSON API +
vanilla-JS front end, no framework) is proven, already deployed by this plan, and
would slot onto the same droplet as a third service with no new infrastructure.
Deploy Streamlit now; rebuild the frontend later as its own project, and swap the
port behind the same subdomain when it's ready.

## 2. Architecture (deliberately boring)

```
phone ──HTTPS──> Caddy (:443, auto-TLS) ──proxy──> uvicorn (:8700) ──> PlantModel
                    │                                    │
              Let's Encrypt cert              model.pt + temperature.json
                (auto-renewed)                     + ood_bank.npz
```

- **Caddy** over nginx: it obtains and renews the Let's Encrypt certificate
  automatically from a 3-line config. nginx needs certbot wired up separately.
- **systemd** keeps uvicorn alive across crashes and reboots (the MPS segfault
  that killed a local session is exactly what a restart policy is for).
- **CPU inference**, one worker. At 0.1 s/request that's ~10 req/s — far beyond
  a personal app's needs. Adding workers multiplies the 394 MB, so don't unless
  traffic actually demands it.

---

## 3. Steps

**A. Provision.** DO droplet, Ubuntu 24.04, **2 GB RAM / 1 vCPU / 50 GB** (~$12/mo),
same SSH key as before. Note the IP.

**B. Point the domain.** Create an `A` record for the domain/subdomain → droplet
IP. (DuckDNS: set the IP in their dashboard.) Verify with `dig +short <domain>`
before installing Caddy — TLS issuance fails if DNS hasn't propagated.

**C. Base setup** (on the droplet, as root):
```
adduser --disabled-password --gecos "" fieldnote
apt update && apt install -y python3-venv python3-pip caddy git
```

**D. Ship the code + model.** Code via git; the model artifacts are gitignored
and must be copied separately (from the Mac):
```
rsync -avz --exclude '.venv' --exclude 'data/images' --exclude 'runs' \
      ~/Desktop/ct-plant-id/ root@<IP>:/opt/fieldnote/
rsync -avz ~/Desktop/ct-plant-id/runs/b_stage2/ root@<IP>:/opt/fieldnote/runs/b_stage2/
```

**E. Install deps** (CPU-only torch — this is the big disk/memory saver):
```
python3 -m venv /opt/fieldnote/.venv
/opt/fieldnote/.venv/bin/pip install -r /opt/fieldnote/requirements-serve.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
chown -R fieldnote:fieldnote /opt/fieldnote
```

**F. Run it as a service** — `deploy/fieldnote.service` (in this repo) →
`/etc/systemd/system/`, then:
```
systemctl daemon-reload && systemctl enable --now fieldnote
systemctl status fieldnote          # confirm running
curl -s localhost:8700/api/health   # expect {"ok":true,"species":2510}
```

**G. TLS + reverse proxy** — `deploy/Caddyfile` → `/etc/caddy/Caddyfile` (edit the
domain), then `systemctl reload caddy`. Caddy fetches the certificate on first
request. Visit `https://<domain>` from the phone.

**H. Verify the PWA end-to-end**: page loads over HTTPS, camera capture opens,
an identification returns, "Add to Home Screen" installs it.

---

## 4. Operating it

- **Logs:** `journalctl -u fieldnote -f`
- **Update the model:** rsync a new `runs/<run>/` up, point `CKPT_PATH` at it,
  `systemctl restart fieldnote`. The temperature + OOD bank are picked up
  automatically from beside the checkpoint — but they are **model-specific**, so
  always ship all three together.
- **Deploying app code:** `git pull` in `/opt/fieldnote` + restart.
- **Service worker caching:** the PWA caches `app.js` aggressively; after a
  frontend change, a phone may serve the old file until its cache is cleared.
  Worth fixing properly (versioned cache / network-first for the app shell)
  before the app has real users — it has already caused confusion twice.

## 5. Costs

| item | monthly |
|------|---------|
| droplet (2 GB, **both apps**) | ~$12 |
| domain (covers both via subdomains) | ~$1 (annualized) |
| DO volume (dataset, keep for Workstream C) | ~$7 |

Co-hosting vs. two separate droplets + two domains saves roughly **$156/yr**.

Note the volume is a *training* cost, not a serving one — it can be destroyed
once no further retrains are planned, but the combined wild+ornamental dataset
on it is what makes the next retrain cheap.
