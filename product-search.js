// product-search.js
// Поиск товара по фото: локально в браузере, через CLIP (image-to-image similarity).

import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/+esm';
import { get, set, entries, clear } from 'https://cdn.jsdelivr.net/npm/idb-keyval@6/+esm';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const DB_PREFIX = 'clip-embed:';

let processorPromise = null;
let modelPromise = null;

function loadModel() {
  if (!processorPromise) processorPromise = AutoProcessor.from_pretrained(MODEL_ID);
  if (!modelPromise) modelPromise = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { quantized: true });
  return Promise.all([processorPromise, modelPromise]);
}

function normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

async function embedImage(image) {
  const [processor, model] = await loadModel();
  let raw;
  try {
    if (typeof image === 'string') {
      // Пробуем загрузить через fetch с обходом CORS
      try {
        raw = await RawImage.fromURL(image);
      } catch (corsError) {
        console.warn('CORS ошибка при загрузке:', image, corsError);
        throw new Error(`Не удалось загрузить изображение: ${image.substring(0, 50)}... (CORS)`);
      }
    } else {
      raw = await RawImage.read(image);
    }
  } catch (err) {
    console.error('Ошибка загрузки изображения:', err);
    throw err;
  }
  
  const inputs = await processor(raw);
  const { image_embeds } = await model(inputs);
  return normalize(image_embeds.data);
}

export async function buildIndex(products, opts = {}) {
  const { onProgress, force = false } = opts;

  const jobs = [];
  for (const p of products) {
    if (!p.code || !Array.isArray(p.images)) continue;
    for (const url of p.images) {
      if (url && url.startsWith('http')) jobs.push({ code: p.code, url });
    }
  }

  let done = 0;
  let indexed = 0;
  const errors = [];

  for (const { code, url } of jobs) {
    const key = DB_PREFIX + url;
    if (!force) {
      const existing = await get(key);
      if (existing) { 
        done++; 
        onProgress?.(done, jobs.length); 
        continue; 
      }
    }
    try {
      const embedding = await embedImage(url);
      await set(key, { code, url, embedding: Array.from(embedding) });
      indexed++;
    } catch (err) {
      errors.push({ url, error: err.message });
      console.warn('Не удалось проиндексировать:', url, err.message);
    }
    done++;
    onProgress?.(done, jobs.length);
  }

  if (errors.length > 0) {
    console.warn(`Индексировано с ошибками: ${errors.length} из ${jobs.length}`);
  }

  return { total: jobs.length, indexed, errors };
}

export async function findSimilar(photo, topN = 5) {
  const queryEmbedding = await embedImage(photo);
  const all = await entries();

  const scored = [];
  for (const [key, value] of all) {
    if (typeof key !== 'string' || !key.startsWith(DB_PREFIX)) continue;
    const embedding = Float32Array.from(value.embedding);
    scored.push({
      code: value.code,
      imageUrl: value.url,
      score: cosine(queryEmbedding, embedding),
    });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topN);
}

export async function getIndexSize() {
  const all = await entries();
  return all.filter(([k]) => typeof k === 'string' && k.startsWith(DB_PREFIX)).length;
}

export async function clearIndex() {
  await clear();
}