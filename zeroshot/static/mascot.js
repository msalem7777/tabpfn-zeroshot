/* Tabby pixel art and poses extracted from Prior Labs. See THIRD_PARTY_ASSETS.md. */
(function () {
  var PAL = { o:'#252A66', b:'#5B68D8', d:'#4A55B8', g:'#74E85E', k:'#1B2046', p:'#E86FA7', t:'#7DE8D8', l:'#AFA9EC', w:'#F5F3FF', a:'#5FC9C0' };
  var LIT = { b:'#7E8BF0', d:'#5663CC', g:'#A0F590', p:'#F29CC6', t:'#A9F2E8', l:'#C7C2F5', w:'#FFFFFF' };
  var DRK = { b:'#4350B0', d:'#39438F', g:'#52BE45', p:'#C2558C', t:'#55BFAE', l:'#8A82CC', w:'#D8D4F0' };
  var SL = { b:'#6873E0', d:'#525EC4', g:'#8BEE78', p:'#EE82B4', t:'#92ECDE', l:'#BBB5F1', w:'#FFFFFF' };
  var SD = { b:'#4E5AC6', d:'#414CA6', g:'#63D24F', p:'#D2609A', t:'#69D2C4', l:'#9C95DC', w:'#E6E2F8' };

  function grid(w, h) {
    var g = [];
    for (var y = 0; y < h; y++) {
      var r = [];
      for (var x = 0; x < w; x++) r.push('.');
      g.push(r);
    }
    return g;
  }
  function P(g, x, y, c) {
    x = Math.round(x); y = Math.round(y);
    if (y >= 0 && y < g.length && x >= 0 && x < g[0].length) g[y][x] = c;
  }
  function R(g, x0, y0, x1, y1, c) {
    for (var y = y0; y <= y1; y++) for (var x = x0; x <= x1; x++) P(g, x, y, c);
  }
  function C(g, cx, cy, r, c, expect) {
    for (var y = Math.floor(cy - r); y <= Math.ceil(cy + r); y++) {
      for (var x = Math.floor(cx - r); x <= Math.ceil(cx + r); x++) {
        var dx = x - cx, dy = y - cy;
        if (dx * dx + dy * dy <= r * r) {
          if (expect === undefined || (g[y] && g[y][x] === expect)) P(g, x, y, c);
        }
      }
    }
  }
  function RR(g, x0, y0, x1, y1, rad, fill, outline) {
    var cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    var hw = (x1 - x0) / 2, hh = (y1 - y0) / 2;
    var r = Math.min(rad, hw, hh);
    function inside(x, y) {
      if (x < x0 || x > x1 || y < y0 || y > y1) return false;
      var qx = Math.max(Math.abs(x - cx) - (hw - r), 0);
      var qy = Math.max(Math.abs(y - cy) - (hh - r), 0);
      return qx * qx + qy * qy <= r * r + 0.3;
    }
    for (var y = y0; y <= y1; y++) {
      for (var x = x0; x <= x1; x++) {
        if (!inside(x, y)) continue;
        var edge = !inside(x - 1, y) || !inside(x + 1, y) || !inside(x, y - 1) || !inside(x, y + 1);
        P(g, x, y, edge ? outline : fill);
      }
    }
  }
  function FILLIF(g, x0, y0, x1, y1, expect, c) {
    for (var y = y0; y <= y1; y++) for (var x = x0; x <= x1; x++) {
      if (g[y] && g[y][x] === expect) g[y][x] = c;
    }
  }
  function face(g, kind) {
    if (kind === 'dizzy') {
      var cs = [23, 33];
      for (var i = 0; i < 2; i++) {
        var cx = cs[i];
        P(g, cx - 1, 17, 'g'); P(g, cx + 1, 17, 'g'); P(g, cx, 18, 'g');
        P(g, cx - 1, 19, 'g'); P(g, cx + 1, 19, 'g');
      }
      R(g, 27, 22, 29, 23, 'g');
      return;
    }
    if (kind === 'happy') {
      var cs2 = [23, 33];
      for (var j = 0; j < 2; j++) {
        var c2 = cs2[j];
        P(g, c2 - 1, 18, 'g'); P(g, c2, 17, 'g'); P(g, c2 + 1, 18, 'g');
      }
      P(g, 25, 21, 'g'); P(g, 31, 21, 'g');
      R(g, 26, 22, 30, 22, 'g');
      return;
    }
    if (kind === 'blink') {
      R(g, 22, 18, 24, 18, 'g'); R(g, 32, 18, 34, 18, 'g');
    } else {
      R(g, 22, 17, 24, 19, 'g'); R(g, 32, 17, 34, 19, 'g');
    }
    P(g, 25, 22, 'g'); P(g, 31, 22, 'g');
    R(g, 26, 23, 30, 23, 'g');
  }

  function buildTabby(o) {
    var g = grid(48, 56);
    RR(g, 14, 46, 20, 52, 2, 'd', 'o');
    RR(g, 12, 50, 23, 55, 2, 'd', 'o');
    if (o.legs === 'idle') {
      RR(g, 28, 46, 34, 52, 2, 'd', 'o');
      RR(g, 26, 50, 37, 55, 2, 'd', 'o');
    } else if (o.legs === 'windup') {
      RR(g, 30, 46, 36, 52, 2, 'd', 'o');
      RR(g, 28, 50, 39, 55, 2, 'd', 'o');
    } else {
      RR(g, 28, 45, 42, 50, 2, 'd', 'o');
      RR(g, 40, 46, 45, 54, 2, 'd', 'o');
    }
    if (o.arms === 'down') RR(g, 39, 32, 44, 44, 2, 'd', 'o');
    else RR(g, 42, 18, 46, 33, 2, 'd', 'o');
    R(g, 29, 1, 30, 5, 'o');
    C(g, 29.5, 1.5, 2.2, 't');
    RR(g, 7, 5, 44, 28, 6, 'b', 'o');
    FILLIF(g, 8, 13, 13, 27, 'b', 'd');
    FILLIF(g, 8, 10, 43, 12, 'b', 'p');
    var dashes = [13, 21, 29, 37];
    for (var di = 0; di < dashes.length; di++) {
      FILLIF(g, dashes[di], 10, dashes[di] + 1, 11, 'p', 'w');
    }
    RR(g, 16, 13, 41, 26, 3, 'k', 'd');
    face(g, o.face);
    C(g, 5.5, 19.5, 3.4, 'o'); C(g, 5.5, 19.5, 2.3, 'p');
    R(g, 24, 29, 31, 30, 'd');
    RR(g, 11, 30, 40, 47, 5, 'b', 'o');
    FILLIF(g, 12, 32, 15, 46, 'b', 'd');
    RR(g, 19, 33, 38, 44, 2, 'k', 'd');
    var xs = [22, 27, 32], c1 = ['t', 'p', 'l'], c2 = ['p', 'g', 't'];
    for (var i2 = 0; i2 < 3; i2++) {
      R(g, xs[i2], 35, xs[i2] + 2, 37, c1[i2]);
      R(g, xs[i2], 40, xs[i2] + 2, 42, c2[i2]);
    }
    if (o.arms === 'down') RR(g, 4, 32, 9, 44, 2, 'd', 'o');
    else RR(g, 2, 18, 6, 33, 2, 'd', 'o');
    return g;
  }

  function makeSprite(rows) {
    var w = rows[0].length, h = rows.length;
    var c = document.createElement('canvas');
    c.width = w; c.height = h;
    var x2 = c.getContext('2d');
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var ch = rows[y][x];
        if (ch === '.' || !PAL[ch]) continue;
        var col = PAL[ch];
        if (LIT[ch]) {
          var s = y; while (s > 0 && rows[s - 1][x] === ch) s--;
          var e2 = y; while (e2 < h - 1 && rows[e2 + 1][x] === ch) e2++;
          var sH = x; while (sH > 0 && rows[y][sH - 1] === ch) sH--;
          var eH = x; while (eH < w - 1 && rows[y][eH + 1] === ch) eH++;
          var lenV = e2 - s + 1, lenH = eH - sH + 1;
          var relV = lenV > 1 ? (y - s) / (lenV - 1) : 0.5;
          var relH = lenH > 1 ? (x - sH) / (lenH - 1) : 0.5;
          if (lenV >= 4) {
            var n = (x * 374761393 + y * 668265263) ^ ((x + 7) * (y + 13) * 1274126177);
            n = ((n ^ (n >> 13)) >>> 0) % 1000 / 1000;
            var m = relV * 0.62 + relH * 0.38 + (n - 0.5) * 0.24;
            if (y === s) col = LIT[ch];
            else if (y === e2) col = (n < 0.3 && (ch === 'b' || ch === 'd')) ? PAL.a : DRK[ch];
            else if (m <= 0.2) col = LIT[ch];
            else if (m <= 0.4) col = SL[ch];
            else if (m < 0.62) col = PAL[ch];
            else if (m < 0.8) col = SD[ch];
            else col = DRK[ch];
            if (col === PAL[ch] && n > 0.96) col = SL[ch];
          } else {
            if (y === s && lenV >= 2) col = LIT[ch];
            else if (y === e2 && lenV >= 3) col = DRK[ch];
          }
          if (ch === 'g') {
            var upN = y > 0 ? rows[y - 1][x] : '.';
            var leftN = x > 0 ? rows[y][x - 1] : '.';
            var rightN = x < w - 1 ? rows[y][x + 1] : '.';
            var downN = y < h - 1 ? rows[y + 1][x] : '.';
            if (upN !== 'g' && leftN !== 'g' && rightN === 'g' && downN === 'g') col = '#ECFFE9';
          }
        }
        x2.fillStyle = col;
        x2.fillRect(x, y, 1, 1);
      }
    }
    return c;
  }

  var SPR = {
    idle: makeSprite(buildTabby({ face: 'idle', legs: 'idle', arms: 'down' })),
    blink: makeSprite(buildTabby({ face: 'blink', legs: 'idle', arms: 'down' })),
    dizzy: makeSprite(buildTabby({ face: 'dizzy', legs: 'idle', arms: 'down' })),
    windup: makeSprite(buildTabby({ face: 'idle', legs: 'windup', arms: 'down' })),
    kick: makeSprite(buildTabby({ face: 'idle', legs: 'kick', arms: 'down' })),
    celeb: makeSprite(buildTabby({ face: 'happy', legs: 'idle', arms: 'up' }))
  };


  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const instances = [...document.querySelectorAll('[data-tabby]')].map(canvas => ({
    canvas, ctx: canvas.getContext('2d'), mode: 'idle', until: 0, blinkAt: 2500,
    blinkUntil: 0, tap: 0, w: 0, h: 0
  }));
  let request = 0, lastFrame = 0, clock = 0;
  function fit(item) {
    const box = item.canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    item.w = box.width; item.h = box.height; item.dpr = dpr;
    item.canvas.width = Math.round(box.width * dpr);
    item.canvas.height = Math.round(box.height * dpr);
  }
  function pose(item, mode, now, duration) {
    item.mode = mode; item.until = now + duration;
    item.canvas.dataset.pose = mode;
  }
  function draw(item, now) {
    if (!item.w || !item.h) return;
    if (now >= item.until && item.mode !== 'idle') {
      if (item.mode === 'windup') pose(item, 'kick', now, 220);
      else pose(item, 'idle', now, Infinity);
    }
    if (now >= item.blinkAt) { item.blinkUntil = now + 130; item.blinkAt = now + 2000 + Math.random() * 3000; }
    const ctx = item.ctx, w = item.w, h = item.h;
    const scale = Math.min(w / 30, h / 36);
    const t = reduced.matches ? 0 : now / 1000;
    // Match the source's hover, sway, celebration bounce and windup shift.
    const hover = Math.sin(t * 1.7) * scale * .9;
    let sway = Math.sin(t * .9 + 1.3) * scale * .45;
    if (item.mode === 'windup') sway -= scale * .9;
    if (item.mode === 'kick') sway += scale * 1.1;
    const bounce = item.mode === 'celeb' ? Math.abs(Math.sin(t * 9)) * scale * 1.4 : 0;
    const x = (w - 24 * scale) / 2 + sway;
    const y = h - 31 * scale + hover - bounce;
    ctx.setTransform(item.dpr, 0, 0, item.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h); ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = 'rgba(12,8,40,.4)'; ctx.beginPath();
    ctx.ellipse(w / 2, h - scale, 7 * scale, scale, 0, 0, Math.PI * 2); ctx.fill();
    const kind = reduced.matches ? 'idle' : item.mode === 'idle' && now < item.blinkUntil ? 'blink' : item.mode;
    ctx.drawImage(SPR[kind], x, y, 24 * scale, 28 * scale);
    // Both feet are complete in buildTabby; no ball overlay or erased pixels.
  }
  function frame(now) {
    request = 0;
    if (document.hidden) return;
    if (now - lastFrame >= 1000 / 24) {
      clock = now; lastFrame = now;
      instances.forEach(item => draw(item, now));
    }
    if (!reduced.matches) request = requestAnimationFrame(frame);
  }
  function resume() {
    cancelAnimationFrame(request); request = 0;
    instances.forEach(item => { fit(item); draw(item, clock); });
    if (!document.hidden && !reduced.matches) request = requestAnimationFrame(frame);
  }
  instances.forEach(item => {
    fit(item); item.canvas.dataset.pose = 'idle';
    item.canvas.addEventListener('click', () => {
      if (reduced.matches) return;
      const mode = ['windup', 'celeb', 'dizzy'][item.tap++ % 3];
      pose(item, mode, performance.now(), mode === 'windup' ? 90 : 1500);
    });
    item.canvas.addEventListener('keydown', event => {
      if (event.code === 'Space' || event.code === 'Enter') { event.preventDefault(); item.canvas.click(); }
    });
    new ResizeObserver(() => { fit(item); draw(item, clock); }).observe(item.canvas);
  });
  document.addEventListener('visibilitychange', resume);
  reduced.addEventListener('change', resume);
  resume();
})();
