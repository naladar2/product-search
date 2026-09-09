// product-search.js
// Поиск товара по фото: локально в браузере, через CLIP (image-to-image similarity).
// Модель качается один раз (~90 МБ, кэшируется браузером) и дальше всё работает офлайн.
//
// Использование:
//   import { buildIndex, findSimilar, getIndexSize, clearIndex } from './product-search.js';
//
//   // products — массив строк из твоей таблицы:
//   // [{ code, name, article, barcode, images: [url1, url2] }, ...]
//   await buildIndex(products, { onProgress: (done, total) => console.log(done, total) });
//
//   // file — File/Blob с фото (например из <input type="file" accept="image/*" capture="environment">)
//   const matches = await findSimilar(file, 5);
//   // matches: [{ code, name, article, barcode, imageUrl, score }] отсортировано по убыванию score (0..1)

import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/+esm';
import { get, set, entries, clear } from 'https://cdn.jsdelivr.net/npm/idb-keyval@6/+esm';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const DB_PREFIX = 'clip-embed:'; // ключ в idb-keyval = DB_PREFIX + imageUrl

// Многие хостинги картинок (в т.ч. catalog-cdn.detmir.st) не отдают CORS-заголовки,
// поэтому fetch с фронтенда напрямую блокируется браузером. wsrv.nl — бесплатный
// image-прокси, который скачивает картинку на своей стороне и отдаёт с корректным CORS.
// Если станет ненадёжно/появятся лимиты — заменить на прокси через свой Apps Script backend.
const IMAGE_PROXY = (url) => `https://wsrv.nl/?url=${encodeURIComponent(url)}&output=jpg`;

let processorPromise = null;
let modelPromise = null;

function loadModel() {
  if (!processorPromise) processorPromise = AutoProcessor.from_pretrained(MODEL_ID);
  if (!modelPromise) modelPromise = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8' });
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

  let raw;
  if (typeof image === 'string') {
    raw = await RawImage.fromURL(IMAGE_PROXY(image));
  } else if (image instanceof Blob) {
    // RawImage.read() не умеет File/Blob напрямую — надёжнее через blob-ссылку,
    // тем же путём, что уже проверенно работает для удалённых картинок.
    const objectUrl = URL.createObjectURL(image);
    try {
      raw = await RawImage.fromURL(objectUrl);
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  } else {
    raw = await RawImage.read(image); // HTMLImageElement/canvas и т.п.
  }

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
      if (url) jobs.push({ code: p.code, name: p.name, article: p.article, barcode: p.barcode, url });
    }
  }

  let done = 0;
  let indexed = 0;
  for (const { code, name, article, barcode, url } of jobs) {
    const key = DB_PREFIX + url;
    const existing = force ? null : await get(key);
    if (existing) {
      // Эмбеддинг не пересчитываем (дорого), но метаданные могли обновиться в таблице —
      // подтягиваем их без повторного скачивания и прогона через CLIP.
      if (existing.code !== code || existing.name !== name || existing.article !== article || existing.barcode !== barcode) {
        await set(key, { ...existing, code, name, article, barcode });
      }
      done++;
      onProgress?.(done, jobs.length);
      continue;
    }
    try {
      const embedding = await embedImage(url);
      await set(key, { code, name, article, barcode, url, embedding: Array.from(embedding) });
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
      name: value.name,
      article: value.article,
      barcode: value.barcode,
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

/** Экспортирует весь индекс (эмбеддинги) в JSON-строку — для переноса на другое устройство. */
export async function exportIndex() {
  const all = await entries();
  const data = all
    .filter(([k]) => typeof k === 'string' && k.startsWith(DB_PREFIX))
    .map(([key, value]) => ({ key, value }));
  return JSON.stringify(data);
}

/** Импортирует JSON, полученный из exportIndex(), без пересчёта эмбеддингов. */
export async function importIndex(jsonText) {
  const data = JSON.parse(jsonText);
  for (const { key, value } of data) {
    await set(key, value);
  }
  return data.length;
}
