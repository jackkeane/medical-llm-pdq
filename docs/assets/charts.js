/* Tiny dependency-free SVG chart renderer for the walkthrough pages.
   Marks: 2px lines, >=8px markers with 2px surface rings, bars <=24px with a
   4px rounded data-end, hairline solid grid. Every chart re-renders on resize
   and ships a hover/focus tooltip; values are also in each figure's table view. */

(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var renderers = [];

  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function fmtDefault(v) {
    return Math.abs(v) >= 1000 ? v.toLocaleString("en-US") : String(Math.round(v * 1000) / 1000);
  }

  /* ---------- tooltip ---------- */

  function makeTooltip(plot) {
    var tt = document.createElement("div");
    tt.className = "tooltip";
    plot.appendChild(tt);
    return {
      node: tt,
      show: function (head, rows, px, py) {
        while (tt.firstChild) tt.removeChild(tt.firstChild);
        var h = document.createElement("div");
        h.className = "tt-head";
        h.textContent = head;
        tt.appendChild(h);
        rows.forEach(function (r) {
          var row = document.createElement("div");
          row.className = "tt-row";
          var key = document.createElement("i");
          key.style.background = r.color;
          row.appendChild(key);
          var v = document.createElement("span");
          v.className = "v";
          v.textContent = r.value;
          row.appendChild(v);
          var n = document.createElement("span");
          n.className = "n";
          n.textContent = r.name;
          row.appendChild(n);
          tt.appendChild(row);
        });
        tt.style.display = "block";
        var pw = plot.clientWidth, tw = tt.offsetWidth, th = tt.offsetHeight;
        var x = px + 14;
        if (x + tw > pw - 4) x = px - tw - 14;
        var y = Math.max(4, Math.min(py - th / 2, plot.clientHeight - th - 4));
        tt.style.left = x + "px";
        tt.style.top = y + "px";
      },
      hide: function () { tt.style.display = "none"; }
    };
  }

  /* ---------- line chart ---------- */

  function lineChart(container, cfg) {
    function render() {
      var plot = container.querySelector(".plot");
      while (plot.firstChild) plot.removeChild(plot.firstChild);

      var W = Math.max(320, plot.clientWidth);
      var H = cfg.height || 260;
      var m = { top: (cfg.vLines && cfg.vLines.length) ? 22 : 14, right: cfg.endLabels ? 96 : 20, bottom: 40, left: 48 };
      var iw = W - m.left - m.right, ih = H - m.top - m.bottom;

      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", tabindex: "0" }, plot);
      if (cfg.ariaLabel) svg.setAttribute("aria-label", cfg.ariaLabel);
      svg.style.outline = "none";

      var xs = cfg.xDomain, ys = cfg.yDomain;
      function X(v) { return m.left + (v - xs[0]) / (xs[1] - xs[0]) * iw; }
      function Y(v) { return m.top + ih - (v - ys[0]) / (ys[1] - ys[0]) * ih; }

      // grid + y ticks
      (cfg.yTicks || []).forEach(function (t) {
        el("line", { x1: m.left, x2: m.left + iw, y1: Y(t), y2: Y(t), stroke: "var(--grid)", "stroke-width": 1 }, svg);
        el("text", { x: m.left - 8, y: Y(t) + 4, "text-anchor": "end", fill: "var(--text-muted)", "font-size": 11.5, style: "font-variant-numeric:tabular-nums" }, svg)
          .textContent = (cfg.yFmt || fmtDefault)(t);
      });
      // baseline + x ticks
      el("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih, stroke: "var(--baseline)", "stroke-width": 1 }, svg);
      (cfg.xTicks || []).forEach(function (t) {
        el("text", { x: X(t), y: m.top + ih + 17, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 11.5, style: "font-variant-numeric:tabular-nums" }, svg)
          .textContent = (cfg.xFmt || String)(t);
      });
      if (cfg.xLabel)
        el("text", { x: m.left + iw / 2, y: H - 4, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 11.5 }, svg)
          .textContent = cfg.xLabel;

      // vertical reference lines (e.g. epoch boundaries) — hairline, labeled at top
      (cfg.vLines || []).forEach(function (v) {
        el("line", { x1: X(v.x), x2: X(v.x), y1: m.top, y2: m.top + ih, stroke: "var(--grid)", "stroke-width": 1 }, svg);
        if (v.label)
          el("text", { x: X(v.x), y: m.top - 2, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 11 }, svg)
            .textContent = v.label;
      });

      // series lines
      cfg.series.forEach(function (s) {
        var d = s.points.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1); }).join("");
        if (s.points.length > 1)
          el("path", { d: d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
        if (s.markers) s.points.forEach(function (p) {
          el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 6, fill: "var(--surface)" }, svg);      // 2px surface ring
          el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: s.color }, svg);
        });
      });

      // selective direct end labels
      if (cfg.endLabels) cfg.series.forEach(function (s) {
        var last = s.points[s.points.length - 1];
        var g = el("text", { x: X(last[0]) + 10, y: Y(last[1]) + 4, "text-anchor": "start", "font-size": 12 }, svg);
        var t1 = el("tspan", { fill: "var(--text-primary)", "font-weight": 650 }, g);
        t1.textContent = (cfg.yFmt || fmtDefault)(last[1]);
        var t2 = el("tspan", { fill: "var(--text-secondary)", dx: 4 }, g);
        t2.textContent = s.name;
        el("circle", { cx: X(last[0]), cy: Y(last[1]), r: 6, fill: "var(--surface)" }, svg);
        el("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4, fill: s.color }, svg);
      });

      // vertical gap bracket between two y-values at one x (e.g. train/eval gap)
      (cfg.brackets || []).forEach(function (b) {
        var bx = X(b.x) + (b.dx || 0);
        el("line", { x1: bx, x2: bx, y1: Y(b.y1), y2: Y(b.y2), stroke: "var(--text-muted)", "stroke-width": 1.5 }, svg);
        [b.y1, b.y2].forEach(function (yv) {
          el("line", { x1: bx - 4, x2: bx + 4, y1: Y(yv), y2: Y(yv), stroke: "var(--text-muted)", "stroke-width": 1.5 }, svg);
        });
        var left = b.side === "left";
        var t = el("text", { x: bx + (left ? -8 : 8), y: (Y(b.y1) + Y(b.y2)) / 2 + 4, "text-anchor": left ? "end" : "start", fill: "var(--text-primary)", "font-size": 12, "font-weight": 650 }, svg);
        t.textContent = b.label;
      });

      // annotations
      (cfg.annotations || []).forEach(function (a) {
        var t = el("text", {
          x: X(a.x) + (a.dx || 0), y: Y(a.y) + (a.dy || 0),
          "text-anchor": a.anchor || "start", fill: "var(--text-secondary)", "font-size": 12
        }, svg);
        t.textContent = a.text;
        if (a.line) el("line", {
          x1: X(a.x), y1: Y(a.y) + (a.dy || 0) + 4, x2: X(a.x), y2: Y(a.y) + 6,
          stroke: "var(--baseline)", "stroke-width": 1
        }, svg);
      });

      /* crosshair + tooltip: snaps to the nearest logged step across all series */
      var allX = [];
      cfg.series.forEach(function (s) { s.points.forEach(function (p) { if (allX.indexOf(p[0]) < 0) allX.push(p[0]); }); });
      allX.sort(function (a, b) { return a - b; });

      var cross = el("line", { y1: m.top, y2: m.top + ih, stroke: "var(--baseline)", "stroke-width": 1, visibility: "hidden" }, svg);
      var tt = makeTooltip(plot);
      var idx = -1;

      function showAt(i) {
        idx = i;
        var xv = allX[i];
        cross.setAttribute("x1", X(xv)); cross.setAttribute("x2", X(xv));
        cross.setAttribute("visibility", "visible");
        var rows = [], ySum = 0, yn = 0;
        cfg.series.forEach(function (s) {
          var hit = null;
          s.points.forEach(function (p) { if (p[0] === xv) hit = p; });
          if (hit) {
            rows.push({ color: s.color, value: (cfg.yFmt || fmtDefault)(hit[1]), name: s.name });
            ySum += Y(hit[1]); yn++;
          }
        });
        var rect = plot.getBoundingClientRect();
        var sx = X(xv) / W * rect.width;
        tt.show((cfg.xTooltip || "x ")(xv), rows, sx, yn ? (ySum / yn) / H * rect.height : rect.height / 2);
      }
      function hide() { idx = -1; cross.setAttribute("visibility", "hidden"); tt.hide(); }

      svg.addEventListener("pointermove", function (ev) {
        var rect = svg.getBoundingClientRect();
        var vx = (ev.clientX - rect.left) / rect.width * W;
        var best = 0, bd = 1e9;
        allX.forEach(function (v, i) { var d = Math.abs(X(v) - vx); if (d < bd) { bd = d; best = i; } });
        showAt(best);
      });
      svg.addEventListener("pointerleave", hide);
      svg.addEventListener("focus", function () { showAt(idx < 0 ? allX.length - 1 : idx); });
      svg.addEventListener("blur", hide);
      svg.addEventListener("keydown", function (ev) {
        if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
          ev.preventDefault();
          var i = idx < 0 ? allX.length - 1 : idx + (ev.key === "ArrowRight" ? 1 : -1);
          showAt(Math.max(0, Math.min(allX.length - 1, i)));
        }
        if (ev.key === "Escape") hide();
      });
    }
    renderers.push(render);
    render();
  }

  /* ---------- horizontal bar chart (nominal categories, one hue) ---------- */

  function barChart(container, cfg) {
    function render() {
      var plot = container.querySelector(".plot");
      while (plot.firstChild) plot.removeChild(plot.firstChild);

      var W = Math.max(320, plot.clientWidth);
      var labelW = cfg.labelWidth || 150;
      var barH = 22, rowH = 40;
      var m = { top: 6, right: 84, left: labelW };
      var H = m.top + cfg.bars.length * rowH + 6;
      var iw = W - m.left - m.right;
      var max = cfg.max;
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img" }, plot);
      if (cfg.ariaLabel) svg.setAttribute("aria-label", cfg.ariaLabel);
      var tt = makeTooltip(plot);

      cfg.bars.forEach(function (b, i) {
        var y = m.top + i * rowH + (rowH - barH) / 2;
        var w = Math.max(2, b.value / max * iw);

        el("text", { x: m.left - 10, y: y + barH / 2 + 4, "text-anchor": "end", fill: "var(--text-secondary)", "font-size": 12.5 }, svg)
          .textContent = b.name;

        var r = 4;
        var d = w > r
          ? "M" + m.left + "," + y + " H" + (m.left + w - r) + " Q" + (m.left + w) + "," + y + " " + (m.left + w) + "," + (y + r) +
            " V" + (y + barH - r) + " Q" + (m.left + w) + "," + (y + barH) + " " + (m.left + w - r) + "," + (y + barH) +
            " H" + m.left + " Z"
          : "M" + m.left + "," + y + " h" + w + " v" + barH + " h-" + w + " Z";
        var bar = el("path", { d: d, fill: b.color || "var(--series-1)" }, svg);

        el("text", { x: m.left + w + 8, y: y + barH / 2 + 4, "text-anchor": "start", fill: "var(--text-primary)", "font-weight": 650, "font-size": 12.5, style: "font-variant-numeric:tabular-nums" }, svg)
          .textContent = b.label;

        // full-row transparent hit target, larger than the mark
        var hit = el("rect", { x: 0, y: m.top + i * rowH, width: W, height: rowH, fill: "transparent" }, svg);
        hit.addEventListener("pointermove", function (ev) {
          bar.setAttribute("fill-opacity", "0.82");
          var rect = plot.getBoundingClientRect();
          tt.show(b.name, [{ color: b.color || "var(--series-1)", value: b.label, name: b.tip || "" }],
            (ev.clientX - rect.left), (m.top + i * rowH + rowH / 2) / H * rect.height);
        });
        hit.addEventListener("pointerleave", function () { bar.setAttribute("fill-opacity", "1"); tt.hide(); });
      });

      el("line", { x1: m.left, x2: m.left, y1: 0, y2: H, stroke: "var(--baseline)", "stroke-width": 1 }, svg);
    }
    renderers.push(render);
    render();
  }

  /* ---------- single stacked horizontal bar (part-to-whole) ---------- */

  function stackedBar(container, cfg) {
    function render() {
      var plot = container.querySelector(".plot");
      while (plot.firstChild) plot.removeChild(plot.firstChild);

      var W = Math.max(320, plot.clientWidth);
      var barH = 30, H = barH + (cfg.outsideLabel ? 4 : 0) + 8;
      var m = { left: 0, right: cfg.outsideLabel ? 120 : 0 };
      var iw = W - m.left - m.right;
      var total = 0;
      cfg.segments.forEach(function (s) { total += s.value; });
      var gap = 2;

      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img" }, plot);
      if (cfg.ariaLabel) svg.setAttribute("aria-label", cfg.ariaLabel);
      var tt = makeTooltip(plot);

      var x = m.left;
      cfg.segments.forEach(function (s, i) {
        var w = s.value / total * (iw - gap * (cfg.segments.length - 1));
        var seg = el("rect", { x: x, y: 4, width: Math.max(1.5, w), height: barH, rx: 3, fill: s.color }, svg);

        var label = s.label || (s.name + " · " + fmtDefault(s.value));
        var estimate = label.length * 7.2 + 20;
        if (w > estimate) {
          el("text", { x: x + w / 2, y: 4 + barH / 2 + 4, "text-anchor": "middle", "font-size": 12.5, "font-weight": 600, fill: s.ink || "#ffffff" }, svg)
            .textContent = label;
        } else if (s.labelOutside) {
          el("text", { x: x + w + 8, y: 4 + barH / 2 + 4, "text-anchor": "start", "font-size": 12.5, fill: "var(--text-primary)", "font-weight": 600 }, svg)
            .textContent = label;
        }

        var hit = el("rect", { x: x - gap / 2, y: 0, width: Math.max(24, w + gap), height: H, fill: "transparent" }, svg);
        hit.addEventListener("pointermove", function (ev) {
          seg.setAttribute("fill-opacity", "0.82");
          var rect = plot.getBoundingClientRect();
          tt.show(cfg.title || "", [{ color: s.color, value: s.tipValue || fmtDefault(s.value), name: s.name }],
            (ev.clientX - rect.left), 14);
        });
        hit.addEventListener("pointerleave", function () { seg.setAttribute("fill-opacity", "1"); tt.hide(); });

        x += w + gap;
      });
    }
    renderers.push(render);
    render();
  }

  /* ---------- theme toggle + resize ---------- */

  function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem("ft-theme"); } catch (e) {}
    if (saved === "dark" || saved === "light") document.documentElement.setAttribute("data-theme", saved);
    var btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    function isDark() {
      var mode = document.documentElement.getAttribute("data-theme");
      return mode ? mode === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    function label() {
      var dark = isDark();
      btn.textContent = dark ? "Light mode" : "Dark mode";
      btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    }
    btn.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("ft-theme", next); } catch (e) {}
      label();
    });
    label();
  }

  var rt = null;
  window.addEventListener("resize", function () {
    clearTimeout(rt);
    rt = setTimeout(function () { renderers.forEach(function (r) { r(); }); }, 150);
  });
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTheme);
  } else {
    initTheme();
  }

  window.FT = { lineChart: lineChart, barChart: barChart, stackedBar: stackedBar };
})();
