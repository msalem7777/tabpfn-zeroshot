# Tabby mascot source

Tabby artwork, palette, sprite construction are adapted from Prior Labs:

- Reference: https://platform.priorlabs.ai/football
- Standalone source: https://platform.priorlabs.ai/tabby-football.html
- Retrieved: 2026-10-05

This is the actual procedural canvas artwork served by that page, not an AI-generated recreation or a crop of the screenshot. The upstream response did not include a license grant; no ownership or permission to sublicense the Prior Labs artwork is claimed here. ZeroShot remains a separate project.

The football-only component has been removed. The compact mascot omits the ball overlay; the original robot artwork already contains both complete feet underneath it. No image patch or invented body part is needed.

`zeroshot/static/mascot.js` reuses the original palette, pixel-art construction and idle, blink, windup, kick, celebration and dizzy poses. Its compact logo layout/controller is new integration code: the mascot floats and blinks; clicking or pressing Enter/Space cycles kick, celebration and dizzy animations. Both logo instances share a 24-frame-per-second scheduler that pauses when the tab is hidden. Reduced-motion preferences display the original idle pose without animation.

`zeroshot/static/prior-theme.css` applies the reference dark palette: background `#090a14`, panels `#15171f`, primary `#8078c8`, text `#e8e8ed`, muted text `#8b8d9e`, borders `#ffffff14`, and cyan chart accent `#6db5c4`. The reference's proprietary font is not bundled; system fonts are used.

All mascot code is served locally. It performs no external requests and sends no dataset data or credentials. The reference-page source hash and URLs are recorded in `tabby-source.json`.
