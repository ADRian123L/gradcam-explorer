# Grad-CAM Explorer, NEU steel surface defects

A static page. Nothing runs on a server: the fine-tuned ResNet-18 (INT8 ONNX, 11 MB) runs in the visitor's browser with ONNX Runtime Web, and Grad-CAM is computed from the last feature map and the classifier weights.

Files
- index.html, gradcam.js: the page and the Grad-CAM math
- model.onnx: ResNet-18 fine-tuned on NEU-DET, exported with two outputs (logits, last feature map), INT8 static quantization
- head.json: class names and classifier weights used for the CAM
- samples/, samples.json: 12 validation images, two per class
- vendor/: ONNX Runtime Web 1.19.2 (MIT), vendored so the page has no CDN dependency

Host on GitHub Pages
1. Create a repository (for example gt-ai-industry/gradcam-explorer) and push this folder's contents to the main branch, at the repository root.
2. Repository Settings, Pages, Source: Deploy from a branch, Branch: main, folder: / (root). Save.
3. After a minute the page is live at https://<org-or-user>.github.io/<repo>/.

Any other static host works the same way (Canvas file upload does not, because it serves files one by one without a directory; a GT web space or Netlify does).

Run locally
    python3 -m http.server 8000
then open http://localhost:8000. Opening index.html directly from the file system will not work because browsers block fetch() on file:// URLs.

Regenerate the model
export.py and quant.py in the build notes export resnet18.pt from HW3 with the feature-map output and quantize it with calibration images from the NEU train split. Validation accuracy and class predictions are unchanged by quantization on the 90 images checked; heatmap values move by about 0.03 on average.
