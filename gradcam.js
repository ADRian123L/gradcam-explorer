// Grad-CAM for a ResNet classifier, computed from the last feature map and the
// classifier weights. For a network that ends in global average pooling followed
// by a linear layer, the gradient of the class score with respect to every
// position of feature map k is w[c][k] / (H * W), so the Grad-CAM weights are
// the classifier weights divided by a constant. The constant cancels in the
// min-max normalisation, so the map below is exactly what hook-based Grad-CAM
// returns for the last block (Selvaraju et al., 2017, section 3).

const SIZE = 200;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// rgba: Uint8ClampedArray of SIZE*SIZE*4 pixels. Returns Float32Array in NCHW order.
function preprocess(rgba) {
  const n = SIZE * SIZE;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 3; ch++) {
      out[ch * n + i] = (rgba[i * 4 + ch] / 255 - MEAN[ch]) / STD[ch];
    }
  }
  return out;
}

function softmax(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// features: Float32Array of K*H*W, weights: array of K numbers for the chosen class.
// Returns {map: Float32Array H*W, h, w} after ReLU, before upsampling.
function weightedSum(features, k, h, w, weights) {
  const hw = h * w;
  const map = new Float32Array(hw);
  for (let c = 0; c < k; c++) {
    const wc = weights[c];
    const base = c * hw;
    for (let i = 0; i < hw; i++) map[i] += wc * features[base + i];
  }
  for (let i = 0; i < hw; i++) if (map[i] < 0) map[i] = 0;
  return map;
}

// Bilinear upsampling matching torch F.interpolate(mode="bilinear", align_corners=False).
function upsample(map, h, w, outH, outW) {
  const out = new Float32Array(outH * outW);
  const sy = h / outH;
  const sx = w / outW;
  for (let y = 0; y < outH; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0;
    const y0 = Math.min(Math.floor(fy), h - 1);
    const y1 = Math.min(y0 + 1, h - 1);
    const ly = fy - y0;
    for (let x = 0; x < outW; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0;
      const x0 = Math.min(Math.floor(fx), w - 1);
      const x1 = Math.min(x0 + 1, w - 1);
      const lx = fx - x0;
      const top = map[y0 * w + x0] * (1 - lx) + map[y0 * w + x1] * lx;
      const bot = map[y1 * w + x0] * (1 - lx) + map[y1 * w + x1] * lx;
      out[y * outW + x] = top * (1 - ly) + bot * ly;
    }
  }
  return out;
}

function normalise(map) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of map) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const out = new Float32Array(map.length);
  const range = hi - lo + 1e-8;
  for (let i = 0; i < map.length; i++) out[i] = (map[i] - lo) / range;
  return out;
}

// Full pipeline from the two model outputs to a SIZE x SIZE map in [0, 1].
function gradcam(features, dims, classWeights) {
  const [, k, h, w] = dims;
  const coarse = weightedSum(features, k, h, w, classWeights);
  return normalise(upsample(coarse, h, w, SIZE, SIZE));
}

// Jet colormap, value in [0, 1] -> [r, g, b].
function jet(v) {
  const r = Math.min(Math.max(1.5 - Math.abs(4 * v - 3), 0), 1);
  const g = Math.min(Math.max(1.5 - Math.abs(4 * v - 2), 0), 1);
  const b = Math.min(Math.max(1.5 - Math.abs(4 * v - 1), 0), 1);
  return [r * 255, g * 255, b * 255];
}

if (typeof module !== "undefined") {
  module.exports = { SIZE, preprocess, softmax, gradcam, jet, upsample, weightedSum, normalise };
}
