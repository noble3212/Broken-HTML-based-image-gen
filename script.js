// ==============================================
// script.js — Stable Diffusion Web UI (Frontend)
// ==============================================
// Requires: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
// ==============================================

// --- UI Elements ---
const promptInput = document.getElementById('prompt');
const generateBtn = document.getElementById('generate');
const output = document.getElementById('output');
const spinner = document.getElementById('spinner');
const logEl = document.getElementById('log');

// --- Logging & Spinner ---
function logMessage(msg) {
  console.log(msg);
  if (!logEl) return;
  const el = document.createElement('div');
  el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
}

function setGenerating(on) {
  if (spinner) spinner.style.display = on ? 'inline-block' : 'none';
  if (generateBtn) generateBtn.disabled = !!on;
  logMessage(on ? 'BUSY: generating / loading' : 'IDLE');
}

// --- ONNX Runtime Sessions ---
let textEncoderSession, unetSession, vaeDecoderSession, safetyCheckerSession;

// --- Check for external weights in ONNX file ---
async function hasExternalData(fileUrl) {
  try {
    const res = await fetch(fileUrl);
    if (!res.ok) return false;
    const buffer = await res.arrayBuffer();
    const str = new TextDecoder("utf-8").decode(buffer.slice(0, 1024));
    // If ONNX references external data, the word "external_data" appears
    return /external_data/i.test(str);
  } catch (e) {
    return false;
  }
}

// --- Initialize Models ---
async function initModels() {
  output.innerHTML = "<p>Loading Stable Diffusion ONNX models... please wait ⏳</p>";
  logMessage('initModels: start');
  setGenerating(true);

  try {
    const modelPaths = {
      text: './models/text_encoder/model.onnx',
      unet: './models/unet/model.onnx',
      vae: './models/vae_decoder/model.onnx',
      safety: './models/safety_checker/model.onnx'
    };

    // --- Detect external weights ---
    for (const [name, path] of Object.entries(modelPaths)) {
      const usesExternal = await hasExternalData(path);
      if (usesExternal) {
        logMessage(`❌ Model "${name}" uses external weights (.pb) which are NOT supported in the browser!`);
        output.innerHTML = `<p>❌ Model "${name}" uses external weights (.pb). Convert to a single .onnx file.</p>`;
        setGenerating(false);
        return;
      }
    }

    // --- Load models ---
    const fetchAndCreate = async (path) => {
      const abs = new URL(path, window.location.href).href;
      logMessage('Fetching model: ' + abs);
      const res = await fetch(abs);
      if (!res.ok) {
        throw new Error(`Failed to fetch ${abs}: ${res.status} ${res.statusText}`);
      }
      const buffer = await res.arrayBuffer();
      return ort.InferenceSession.create(buffer);
    };

    const [t, u, v, s] = await Promise.all([
      fetchAndCreate(modelPaths.text),
      fetchAndCreate(modelPaths.unet),
      fetchAndCreate(modelPaths.vae),
      fetchAndCreate(modelPaths.safety)
    ]);

    textEncoderSession = t;
    unetSession = u;
    vaeDecoderSession = v;
    safetyCheckerSession = s;

    const dumpSession = (name, sess) => {
      if (!sess) return logMessage(`${name}: session missing`);
      logMessage(`--- ${name} session ---`);
      logMessage(' inputNames: ' + JSON.stringify(sess.inputNames || []));
      logMessage(' outputNames: ' + JSON.stringify(sess.outputNames || []));
    };
    dumpSession('text encoder', textEncoderSession);
    dumpSession('unet', unetSession);
    dumpSession('vae decoder', vaeDecoderSession);
    dumpSession('safety checker', safetyCheckerSession);

    output.innerHTML = "<p>✅ All models loaded successfully. Ready to generate!</p>";
    logMessage('initModels: models loaded');
  } catch (err) {
    console.error(err);
    logMessage('initModels error: ' + (err && err.message));
    output.innerHTML = "<p>❌ Failed to load one or more ONNX models. Check console/Network tab.</p>";
  } finally {
    setGenerating(false);
  }
}

// --- Tokenizer Stub ---
function tokenizePrompt(prompt) {
  const dims = [1, 77, 768];
  const length = dims.reduce((a,b) => a*b, 1);
  const data = new Float32Array(length);
  for (let i = 0; i < length; i++) data[i] = Math.random();
  return new ort.Tensor('float32', data, dims);
}

// --- Generate Button ---
generateBtn.removeEventListener && generateBtn.removeEventListener('click', ()=>{});
generateBtn.addEventListener('click', async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) return alert('Enter a prompt first!');
  if (!textEncoderSession || !unetSession || !vaeDecoderSession) {
    return alert('Models are still loading...');
  }

  setGenerating(true);
  output.innerHTML = `<p>🔮 Generating image for: <b>${prompt}</b>...</p>`;
  logMessage('generate: start, prompt="' + prompt + '"');

  try {
    // --- Text Encoding ---
    let textEmbeds;
    try {
      const textInputName = textEncoderSession.inputNames[0] || 'input_ids';
      const tok = tokenizePrompt(prompt);
      logMessage('Running text encoder with input: ' + textInputName);
      const textOut = await textEncoderSession.run({ [textInputName]: tok });
      const outKey = textEncoderSession.outputNames[0] || Object.keys(textOut)[0];
      textEmbeds = textOut[outKey];
      logMessage('text encoder output key: ' + outKey + ' dims: ' + JSON.stringify(textEmbeds && textEmbeds.dims));
    } catch (e) {
      logMessage('Text encoder run failed, using fallback embeddings. Error: ' + (e && e.message));
      const dims = [1, 77, 768];
      const buf = new Float32Array(dims.reduce((a,b)=>a*b,1));
      for (let i=0;i<buf.length;i++) buf[i] = Math.random();
      textEmbeds = new ort.Tensor('float32', buf, dims);
      logMessage('Fallback textEmbeds dims: ' + JSON.stringify(textEmbeds.dims));
    }

    // --- Initialize latent ---
    const latentShape = [1, 4, 64, 64];
    const latentBuf = new Float32Array(latentShape.reduce((a,b)=>a*b,1));
    for (let i=0;i<latentBuf.length;i++) latentBuf[i] = (Math.random()*2-1);
    const latent = new ort.Tensor('float32', latentBuf, latentShape);

    // --- Build UNet inputs ---
    const unetInputs = {};
    for (const name of unetSession.inputNames) {
      if (/sample|latent/i.test(name)) unetInputs[name] = latent;
      else if (/encoder|hidden_states|text/i.test(name)) unetInputs[name] = textEmbeds;
      else if (/timestep|timesteps|t\b/i.test(name)) unetInputs[name] = new ort.Tensor('float32', Float32Array.from([1.0]), [1]);
      else {
        const size = latentShape.reduce((a,b)=>a*b,1);
        unetInputs[name] = new ort.Tensor('float32', new Float32Array(size), latentShape);
        logMessage(`WARNING: created fallback for UNet input "${name}"`);
      }
    }
    logMessage('Running UNet with keys: ' + JSON.stringify(Object.keys(unetInputs)));

    const unetOutMap = await unetSession.run(unetInputs);
    const unetOut = unetOutMap[unetSession.outputNames[0]] || unetOutMap[Object.keys(unetOutMap)[0]];
    logMessage('unetOut dims: ' + JSON.stringify(unetOut && unetOut.dims));

    // --- VAE decode ---
    const vaeInName = vaeDecoderSession.inputNames.find(n=>/latent|sample/i.test(n)) || vaeDecoderSession.inputNames[0];
    logMessage('Using VAE input: ' + vaeInName);
    const vaeOutMap = await vaeDecoderSession.run({ [vaeInName]: unetOut });
    const vaeOut = vaeOutMap[vaeDecoderSession.outputNames[0]] || vaeOutMap[Object.keys(vaeOutMap)[0]];
    logMessage('VAE output dims: ' + JSON.stringify(vaeOut && vaeOut.dims));

    if (!vaeOut) throw new Error('VAE decoder returned no tensor');

    // --- Display Image ---
    const img = tensorToImage(vaeOut);
    output.innerHTML = '';
    output.appendChild(img);
    logMessage('Image appended to output');
  } catch (err) {
    console.error('generation error:', err);
    logMessage('generation error: ' + (err && err.message));
    output.innerHTML = `<p>❌ Error during inference: ${err && err.message}</p>`;
  } finally {
    setGenerating(false);
  }
});

// --- Tensor to Image Conversion ---
function tensorToImage(tensor) {
  if (!tensor) throw new Error('tensorToImage: no tensor provided');
  const [b, c, h, w] = tensor.dims;
  const data = tensor.data;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(w, h);

  const plane = w * h;
  for (let i = 0; i < plane; i++) {
    const r = data[i] ?? 0;
    const g = data[i + plane] ?? r;
    const bVal = data[i + 2 * plane] ?? r;

    const toByte = (v) => {
      let x = v;
      if (x < -1 || x > 1) x = Math.max(0, Math.min(1, x));
      else x = (x + 1) / 2;
      return Math.round(Math.max(0, Math.min(255, x * 255)));
    };

    imgData.data[i*4+0] = toByte(r);
    imgData.data[i*4+1] = toByte(g);
    imgData.data[i*4+2] = toByte(bVal);
    imgData.data[i*4+3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
  const img = document.createElement('img');
  img.src = canvas.toDataURL('image/png');
  return img;
}

// --- Initialize Models on Page Load ---
initModels();

// --- Global Error Handling ---
window.addEventListener('error', (e) => {
  console.error('Uncaught error:', e.error || e.message);
  logMessage('Uncaught error: ' + (e.error ? e.error.message : e.message));
});
window.addEventListener('unhandledrejection', (e) => {
  console.error('Unhandled rejection:', e.reason);
  const reasonMsg = e.reason && e.reason.message ? e.reason.message : JSON.stringify(e.reason);
  logMessage('Unhandled rejection: ' + reasonMsg);
});
