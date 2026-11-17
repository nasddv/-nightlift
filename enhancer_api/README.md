# FreqSSM-Lite enhancer: model library + simple interface

```
enhancer_api/
├── index.html        simple interface (upload → Enhance → output → Download)
├── freqssm.js        the model library: code only, no weights, no UI
└── models/
    ├── lol.js        weights: low-light model (LOL-v2, test PSNR 21.0 dB)
    └── sice.js       weights: mixed-exposure model (SICE, test PSNR 16.3 dB)
```

Works offline in any modern browser, also when opened as a file (double-click `index.html`).
Copy the whole folder: `index.html` needs `freqssm.js` and `models/` beside it.

## Use it in any frontend

1. Include the weights you need, then the library:
```html
<script src="models/lol.js"></script>
<script src="freqssm.js"></script>
```
2. Give it an image, get the enhanced image back:
```js
const result = await FreqSSM.enhance(fileOrImage, { model: "lol", maxSide: 512 });

document.body.appendChild(result.canvas);   // show it
const blob = await result.toBlob();         // PNG Blob, e.g. to upload or download
const url  = result.toDataURL();            // or a data: URL for an <img src>
```

### `FreqSSM.enhance(source, options)` → `Promise<result>`

| | |
|---|---|
| `source` | `File`, `Blob`, `<img>`, `<canvas>`, `ImageBitmap`, `ImageData`, or an image URL |
| `options.model` | `"lol"` (dark photos, default) or `"sice"` (mixed exposure) |
| `options.maxSide` | longest side processed, default `512`; `0` = original size (slower) |
| `options.onProgress` | `function (fraction 0..1)` |
| `result` | `{ canvas, imageData, width, height, ms, toBlob(type), toDataURL(type) }` |

Other calls: `FreqSSM.models()` lists the loaded models; `FreqSSM.registerModel(name, weights)` adds weights
when you import them yourself (e.g. with a bundler).

The model runs in a background Web Worker, so the page stays responsive. Calls are processed one at a time.

### React example
```jsx
// add the two <script> tags to public/index.html, then:
async function onFile(e) {
  const r = await window.FreqSSM.enhance(e.target.files[0], { model: "lol" });
  setOutputUrl(r.toDataURL());
}
<input type="file" accept="image/*" onChange={onFile} />
<img src={outputUrl} />
```
