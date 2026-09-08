// product-search.js
// Поиск товара по фото: локально в браузере, через CLIP (image-to-image similarity).
// Модель качается один раз (~90 МБ, кэшируется браузером) и дальше всё работает офлайн.
//
// Использование:
//   import { buildIndex, findSimilar, getIndexSize, clearIndex } from './product-search.js';
//
//   // products — массив строк из твоей таблицы: [{ code, images: [url1, url2] }, ...]
//   // code = колонка D, images = ссылки из колонок J и K (пустые/битые пропускаются)
//   await buildIndex(products, { onProgress: (done, total) => console.log(done, total) });
//
//   // file — File/Blob с фото (например из <input type="file" accept="image/*" capture="environment">)
//   const matches = await findSimilar(file, 5);
//   // matches: [{ code, imageUrl, score }] отсортировано по убыванию score (0..1)

import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/+esm';
import { get, set, entries, clear } from 'https://cdn.jsdelivr.net/npm/idb-keyval@6/+esm';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const DB_PREFIX = 'clip-embed:'; // ключ в idb-keyval = DB_PREFIX + imageUrl

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
  return dot; // оба вектора уже нормализованы -> dot = cosine similarity
}

// image: URL-строка, File/Blob, HTMLImageElement/HTMLCanvasElement
async function embedImage(image) {
  const [processor, model] = await loadModel();
  const raw = typeof image === 'string'
    ? await RawImage.fromURL(image)
    : await RawImage.read(image);
  const inputs = await processor(raw);
  const { image_embeds } = await model(inputs);
  return normalize(image_embeds.data);
}

/**
 * Строит/обновляет индекс эмбеддингов для каталога.
 * Уже проиндексированные (по url) картинки повторно не считаются — можно звать при каждом запуске.
 * @param {{code:string, images:string[]}[]} products
 * @param {{onProgress?:(done:number,total:number)=>void, force?:boolean}} opts
 */
export async function buildIndex(products, opts = {}) {
  const { onProgress, force = false } = opts;

  const jobs = [];
  for (const p of products) {
    if (!p.code || !Array.isArray(p.images)) continue;
    for (const url of p.images) {
      if (url) jobs.push({ code: p.code, url });
    }
  }

  let done = 0;
  let indexed = 0;
  for (const { code, url } of jobs) {
    const key = DB_PREFIX + url;
    if (!force) {
      const existing = await get(key);
      if (existing) { done++; onProgress?.(done, jobs.length); continue; }
    }
    try {
      const embedding = await embedImage(url);
      await set(key, { code, url, embedding: Array.from(embedding) });
      indexed++;
    } catch (err) {
      // Частая причина — CORS на хостинге картинки, или битая ссылка. Пропускаем и едем дальше.
      console.warn('Не удалось проиндексировать', url, err);
    }
    done++;
    onProgress?.(done, jobs.length);
  }
  return { total: jobs.length, indexed };
}

/**
 * Ищет похожие товары по фото.
 * @param {File|Blob|string|HTMLImageElement} photo
 * @param {number} topN
 * @returns {Promise<{code:string, imageUrl:string, score:number}[]>}
 */
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
