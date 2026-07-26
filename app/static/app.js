/* ===========================================================================
   Fieldnote — CT Plant ID  (vanilla, no build step, no dependencies)

   Flow:  start  ->  busy (preview + call /api/identify)  ->  results | error
   All server strings are inserted via textContent / DOM APIs — never innerHTML
   with untrusted data — so species and common names can't inject markup.
   =========================================================================== */
(function () {
  "use strict";

  var API_IDENTIFY = "/api/identify";
  var API_HEALTH = "/api/health";

  // ---- element handles -----------------------------------------------------
  var views = {
    start:   document.getElementById("viewStart"),
    busy:    document.getElementById("viewBusy"),
    results: document.getElementById("viewResults"),
    error:   document.getElementById("viewError"),
  };
  var fileInput   = document.getElementById("fileInput");
  var libraryBtn  = document.getElementById("libraryBtn");
  var previewImg  = document.getElementById("previewImg");
  var resultImg   = document.getElementById("resultImg");
  var cancelBtn   = document.getElementById("cancelBtn");
  var againBtn    = document.getElementById("againBtn");
  var retryBtn    = document.getElementById("retryBtn");
  var notSureBanner = document.getElementById("notSureBanner");
  var topCard     = document.getElementById("topCard");
  var candidatesEl = document.getElementById("candidates");
  var othersHeading = document.getElementById("othersHeading");
  var errorTitle  = document.getElementById("errorTitle");
  var errorMsg    = document.getElementById("errorMsg");

  var currentObjectURL = null;
  var inFlight = null; // AbortController for the active request

  // ---- view switching ------------------------------------------------------
  function show(name) {
    Object.keys(views).forEach(function (k) {
      var hidden = k !== name;
      views[k].hidden = hidden;
    });
    // Scroll back to top on view change so results start at the photo.
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  // ---- small helpers -------------------------------------------------------
  function el(tag, className, text) {
    var n = document.createElement(tag);
    if (className) n.className = className;
    if (text != null) n.textContent = text;
    return n;
  }

  // Title-case a common name if it arrives lowercase ("garlic mustard" ->
  // "Garlic Mustard"). Only capitalize at a word start (string start, space,
  // or hyphen) so apostrophes stay intact ("dame's rocket" -> "Dame's Rocket").
  function titleCase(s) {
    return s.replace(/(^|[\s-])([a-z])/g, function (m, sep, c) { return sep + c.toUpperCase(); });
  }

  // Decide the display name pair. common_name may be null -> use species.
  function names(cand) {
    var latin = (cand.species || "").trim();
    var common = cand.common_name ? titleCase(String(cand.common_name).trim()) : "";
    if (common) return { primary: common, secondary: latin, hasLatin: true };
    return { primary: latin || "Unknown plant", secondary: "", hasLatin: false };
  }

  function pct(prob) {
    var p = Math.max(0, Math.min(1, Number(prob) || 0));
    return Math.round(p * 100);
  }

  // Map status + is_weed -> a badge {cls, label, warn}. Invasive/weed is loud.
  function badgeInfo(cand) {
    var status = (cand.status || "unknown").toLowerCase();
    var weed = !!cand.is_weed;
    if (status === "invasive") {
      return { cls: "invasive", label: weed ? "Invasive weed" : "Invasive", warn: true };
    }
    if (weed) {
      // A weed that isn't formally invasive (often an introduced nuisance).
      return { cls: "weed", label: "Weed", warn: true };
    }
    if (status === "native")     return { cls: "native", label: "Native", warn: false };
    if (status === "introduced") return { cls: "introduced", label: "Introduced", warn: false };
    // A cultivated garden/landscape plant — useful precisely because it means
    // "planted on purpose", i.e. not something to pull.
    if (status === "ornamental") return { cls: "ornamental", label: "Garden plant", warn: false };
    return { cls: "unknown", label: "Status unknown", warn: false };
  }

  var WARN_SVG = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" aria-hidden="true"><path d="M12 4l8.5 14.5H3.5L12 4Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v3.4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="16.4" r="1" fill="currentColor"/></svg>';

  function makeBadge(cand, small) {
    var info = badgeInfo(cand);
    var b = el("span", "badge " + info.cls + (small ? " badge-sm" : ""));
    if (info.warn) {
      // Warning triangle for invasive/weed (built as SVG, not from server data).
      var wrap = document.createElement("span");
      wrap.setAttribute("aria-hidden", "true");
      wrap.style.display = "inline-flex";
      wrap.innerHTML = WARN_SVG; // static, author-controlled markup only
      b.appendChild(wrap);
    } else {
      b.appendChild(el("span", "glyph"));
    }
    b.appendChild(document.createTextNode(info.label));
    return b;
  }

  // ---- render results ------------------------------------------------------
  function renderResults(data) {
    var cands = Array.isArray(data.candidates) ? data.candidates.slice(0, 5) : [];
    if (cands.length === 0) {
      showError("No match found", "The model didn't return any candidates for that photo. Try a clearer, closer shot of a single plant.");
      return;
    }

    // Caution banner: out-of-scope (OOD) takes priority over low-confidence.
    var oob = data.out_of_scope === true;
    if (oob || data.not_sure) {
      var bStrong = notSureBanner.querySelector("strong");
      var bSpan = notSureBanner.querySelector("span");
      if (oob) {
        bStrong.textContent = "This looks outside my range.";
        bSpan.textContent = "Maybe a garden ornamental or something I wasn't trained on. The closest Connecticut-wild matches are below — treat them with caution.";
      } else {
        bStrong.textContent = "Not confident on this one.";
        bSpan.textContent = "Here are the closest matches — compare them yourself before deciding.";
      }
      notSureBanner.hidden = false;
    } else {
      notSureBanner.hidden = true;
    }

    // ---- top card ----
    topCard.innerHTML = "";
    var top = cands[0];
    var topInfo = badgeInfo(top);
    // Rail color tracks the winner's status so a red edge signals a weed early —
    // but only when we're actually showing status (hidden on low-conf/OOD, so the
    // rail shouldn't flash red there either).
    var showStatus = data.show_status !== false;
    var railVar = (showStatus && topInfo.warn) ? "var(--invasive)"
                : (showStatus && topInfo.cls === "native") ? "var(--native)"
                : (showStatus && topInfo.cls === "introduced") ? "var(--introduced)"
                : (showStatus && topInfo.cls === "ornamental") ? "var(--ornamental)"
                : (oob || data.not_sure) ? "var(--introduced)" : "var(--accent)";
    topCard.style.setProperty("--rail", railVar);

    var rank = el("div", "top-rank");
    rank.appendChild(el("span", "dot"));
    // Lead with the honest, calibrated confidence label ("Strong match" /
    // "Likely match" / "Possible match" / "Uncertain") rather than a raw % —
    // the model is underconfident so the number alone understates reliability.
    // When out-of-scope, don't claim a confidence label ("Strong match" would
    // contradict "outside my range") — call it the closest match instead.
    rank.appendChild(document.createTextNode(
      oob ? "Closest match"
          : (data.lead_with_genus === true) ? "Narrowed to a group"
          : (data.confidence_label || (data.not_sure ? "Best guess" : "Most likely match"))));
    topCard.appendChild(rank);

    // Genus fallback: when the species is a coin-flip but the genus isn't, the
    // genus IS the answer — "an oak" beats a confidently wrong oak. The species
    // guesses remain visible below as possibilities.
    var nm = names(top);
    if (data.lead_with_genus === true && data.genus) {
      var gLabel = data.genus_label || data.genus;
      topCard.appendChild(el("h2", "top-name", "Some kind of " + gLabel));
      topCard.appendChild(el("p", "top-latin", data.genus));
      var gnote = el("p", "genus-note");
      gnote.appendChild(document.createTextNode(
        "Confident about the " + gLabel + ", not the exact species — likely "
        + nm.primary + "."));
      topCard.appendChild(gnote);
    } else {
      topCard.appendChild(el("h2", "top-name", nm.primary));
      if (nm.secondary) topCard.appendChild(el("p", "top-latin", nm.secondary));
    }

    // Hazard warning — independent of status, and shown even when the status
    // flag is suppressed: "will this hurt me" outranks "is it native".
    if (top.hazard) {
      var hz = el("div", "hazard");
      var hzIcon = document.createElement("span");
      hzIcon.setAttribute("aria-hidden", "true");
      hzIcon.style.display = "inline-flex";
      hzIcon.innerHTML = WARN_SVG; // static, author-controlled markup
      hz.appendChild(hzIcon);
      hz.appendChild(el("span", "hazard-text", top.hazard));
      topCard.appendChild(hz);
    }

    var badges = el("div", "top-badges");
    if (data.show_status !== false) {
      badges.appendChild(makeBadge(top, false));
    } else {
      // Safety: hide the native/invasive/WEED flag when we're not confident in
      // the ID. A wrong "Invasive weed" on an out-of-scope plant (e.g. a garden
      // hydrangea) could get a real plant pulled — only show status on a
      // Likely/Strong match. (Reuses the neutral 'unknown' badge styling.)
      var hb = el("span", "badge unknown");
      hb.appendChild(el("span", "glyph"));
      hb.appendChild(document.createTextNode("Status hidden — not confident enough"));
      badges.appendChild(hb);
    }
    topCard.appendChild(badges);

    // Confidence meter. When the headline IS the genus, the number must be the
    // genus probability — showing the species' 40% under "Some kind of oak"
    // would understate exactly the claim being made.
    var leadingGenus = data.lead_with_genus === true && data.genus;
    var confValue = leadingGenus ? (Number(data.genus_prob) || 0) : top.prob;
    var conf = el("div", "confidence");
    var row = el("div", "confidence-row");
    row.appendChild(el("span", "confidence-label",
      leadingGenus ? "Confidence in the group" : "Confidence"));
    row.appendChild(el("span", "confidence-val", pct(confValue) + "%"));
    conf.appendChild(row);
    var meter = el("div", "meter");
    var fill = el("div", "meter-fill" + ((data.not_sure && !leadingGenus) ? " low" : ""));
    meter.appendChild(fill);
    conf.appendChild(meter);
    topCard.appendChild(conf);
    // animate width after paint
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { fill.style.width = pct(confValue) + "%"; });
    });

    // ---- ranked list 2..5 ----
    candidatesEl.innerHTML = "";
    var others = cands.slice(1);
    othersHeading.hidden = others.length === 0;
    others.forEach(function (c, i) {
      var li = el("li", "cand");

      li.appendChild(el("div", "cand-rank", String(i + 2)));

      var body = el("div", "cand-body");
      var cn = names(c);
      body.appendChild(el("div", "cand-name", cn.primary));
      if (cn.secondary) body.appendChild(el("div", "cand-latin", cn.secondary));

      var meta = el("div", "cand-meta");
      var prob = el("span", "cand-prob");
      var mini = el("span", "mini-meter");
      var miniFill = el("i");
      miniFill.style.width = pct(c.prob) + "%";
      mini.appendChild(miniFill);
      prob.appendChild(mini);
      var b = document.createElement("b");
      b.textContent = pct(c.prob) + "%";
      prob.appendChild(b);
      meta.appendChild(prob);
      body.appendChild(meta);
      li.appendChild(body);

      var right = el("div", "cand-right");
      right.appendChild(makeBadge(c, true));
      li.appendChild(right);

      candidatesEl.appendChild(li);
    });

    show("results");
  }

  function showError(title, msg) {
    errorTitle.textContent = title || "Something went wrong";
    errorMsg.textContent = msg || "We couldn't identify that photo. Check your connection and try again.";
    show("error");
  }

  // ---- the request ---------------------------------------------------------
  function identify(file) {
    if (!file) return;
    if (!/^image\//.test(file.type) && file.type !== "") {
      showError("That's not a photo", "Please choose an image file (JPEG or PNG) of a plant.");
      return;
    }

    // preview
    revokeURL();
    currentObjectURL = URL.createObjectURL(file);
    previewImg.src = currentObjectURL;
    resultImg.src = currentObjectURL;
    show("busy");

    var form = new FormData();
    form.append("image", file, file.name || "photo.jpg");

    inFlight = new AbortController();
    var timeout = setTimeout(function () { if (inFlight) inFlight.abort("timeout"); }, 45000);

    fetch(API_IDENTIFY, { method: "POST", body: form, signal: inFlight.signal })
      .then(function (res) {
        return res.text().then(function (text) {
          var body = null;
          try { body = text ? JSON.parse(text) : null; } catch (e) { body = null; }
          if (!res.ok) {
            var detail = body && body.detail ? String(body.detail) : null;
            var e = new Error(detail || ("Server error " + res.status));
            e.httpStatus = res.status;
            e.friendly = detail;
            throw e;
          }
          return body;
        });
      })
      .then(function (data) {
        clearTimeout(timeout);
        inFlight = null;
        if (!data || !Array.isArray(data.candidates)) {
          showError("Unexpected response", "The server replied in a form we didn't understand. Please try again.");
          return;
        }
        renderResults(data);
      })
      .catch(function (err) {
        clearTimeout(timeout);
        if (inFlight === null && err && err.name === "AbortError") return; // user cancelled
        inFlight = null;
        if (err && (err.name === "AbortError" || err.aborted === "timeout")) {
          showError("Taking too long", "The identification timed out. The server may be waking up — please try again.");
          return;
        }
        if (err && err.friendly) {
          // Backend gave a clean 400 (bad/large/empty image) — show it plainly.
          showError("Couldn't read that image", capitalize(err.friendly));
          return;
        }
        showError("Can't reach the server", "We couldn't connect to identify your plant. Check your connection, then try again.");
      });
  }

  function capitalize(s) { s = String(s); return s.charAt(0).toUpperCase() + s.slice(1); }

  function revokeURL() {
    if (currentObjectURL) { URL.revokeObjectURL(currentObjectURL); currentObjectURL = null; }
  }

  function reset() {
    if (inFlight) { var c = inFlight; inFlight = null; try { c.abort(); } catch (e) {} }
    fileInput.value = "";
    show("start");
  }

  // ---- events --------------------------------------------------------------
  fileInput.addEventListener("change", function () {
    var f = fileInput.files && fileInput.files[0];
    if (f) identify(f);
  });

  // "Choose from library" — same input without the camera hint so it opens the picker.
  libraryBtn.addEventListener("click", function () {
    fileInput.removeAttribute("capture");
    fileInput.click();
    // restore camera-first behavior for the primary button next time
    setTimeout(function () { fileInput.setAttribute("capture", "environment"); }, 300);
  });

  cancelBtn.addEventListener("click", reset);
  againBtn.addEventListener("click", reset);
  retryBtn.addEventListener("click", reset);

  window.addEventListener("pagehide", revokeURL);

  // ---- health check --------------------------------------------------------
  function healthCheck() {
    var wrap = document.getElementById("health");
    var label = document.getElementById("healthLabel");
    var foot = document.getElementById("footSpecies");
    fetch(API_HEALTH, { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
      .then(function (d) {
        if (d && d.ok) {
          wrap.classList.remove("down"); wrap.classList.add("ok");
          var n = Number(d.species) || 0;
          label.textContent = "Ready";
          wrap.title = "Model ready — " + n.toLocaleString() + " species";
          if (n > 0 && foot) foot.textContent = "Covers " + n.toLocaleString() + " Connecticut species";
        } else { throw new Error(); }
      })
      .catch(function () {
        wrap.classList.remove("ok"); wrap.classList.add("down");
        label.textContent = "Offline";
        wrap.title = "Can't reach the model server";
      });
  }
  healthCheck();

  // ---- service worker (app-shell only; never caches API) -------------------
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () {
      navigator.serviceWorker.register("/static/sw.js").catch(function () { /* offline shell is a nicety, not required */ });
    });
  }
})();
