const express = require('express');
const cors = require('cors');
const ort = require('onnxruntime-node');

const app = express();
app.use(cors());
app.use(express.json({ limit: '100mb' }));

const MODEL_PATHS = {
  text: 'models/text_encoder/model.onnx',
  unet: 'models/unet/model.onnx',
  vae: 'models/vae_decoder/model.onnx',
  safety: 'models/safety_checker/model.onnx'
};

const sessions = {};

async function loadSessions() {
  for (const [key, path] of Object.entries(MODEL_PATHS)) {
    try {
      sessions[key] = await ort.InferenceSession.create(path, { executionProviders: ['cpu'] });
      console.log(`Loaded ${key} -> ${path}`);
    } catch (e) {
      console.error(`Failed to load ${key}:`, e && e.message);
      sessions[key] = { error: String(e) };
    }
  }
}

app.get('/status', async (req, res) => {
  const info = {};
  for (const [k, sess] of Object.entries(sessions)) {
    if (!sess || sess.error) {
      info[k] = { ok: false, error: sess && sess.error ? sess.error : 'not loaded' };
      continue;
    }
    try {
      const inputs = sess.inputNames || sess.getInputs().map(i => ({ name: i.name, shape: i.shape, type: i.type }));
      const outputs = sess.outputNames || sess.getOutputs().map(o => ({ name: o.name, shape: o.shape, type: o.type }));
      info[k] = { ok: true, inputs, outputs };
    } catch (e) {
      info[k] = { ok: false, error: String(e) };
    }
  }
  res.json(info);
});

const PORT = 5000;
loadSessions().then(() => {
  app.listen(PORT, '127.0.0.1', () => {
    console.log(`Server running on http://127.0.0.1:${PORT}`);
  });
});