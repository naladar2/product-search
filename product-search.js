// product-search.js
// Поиск товара по фото: локально в браузере, через CLIP (image-to-image similarity).

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/+esm';

// Настройка окружения для работы без eval
env.allowLocalModels = false;
env.useBrowserCache = true;

// Используем pipeline вместо прямого вызова модели
let extractor = null;

async function getExtractor() {
  if (!extractor) {
    try {
      // Используем pipeline для создания feature extractor
      extractor = await pipeline('image-feature-extraction', 'Xenova/clip-vit-base-patch32', {
        quantized: true,
        // Важно: не использовать прогресс для упрощения
        progress_callback: null
      });
    } catch (err) {
      console.error('Ошибка загрузки модели:', err);
      throw err;
    }
  }
  return extractor;
}

function normalize(vec) {
  if (!vec || !vec.length) return new Float32Array(0);
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

async function embedImage(image) {
  const extractor = await getExtractor();
  
  let imageData;
  if (typeof image === 'string') {
    // Для URL используем fetch с обходом CORS
    const response = await fetch(image);
    const blob = await response.blob();
    imageData = blob;
  } else {
    imageData = image;
  }
  
  // Получаем эмбеддинг через pipeline
  const result = await extractor(imageData, {
    pooling: 'mean',
    normalize: true
  });
  
  // result - это массив с эмбеддингом
  return result;
}

const DB_PREFIX = 'clip-embed:';

// Используем IndexedDB напрямую
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('ClipIndexDB', 1);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('embeddings')) {
        db.createObjectStore('embeddings', { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getEmbedding(key) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('embeddings', 'readonly');
    const store = tx.objectStore('embeddings');
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function setEmbedding(key, value) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('embeddings', 'readwrite');
    const store = tx.objectStore('embeddings');
    const request = store.put({ id: key, ...value });
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

async function getAllEmbeddings() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('embeddings', 'readonly');
    const store = tx.objectStore('embeddings');
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function clearAll() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('embeddings', 'readwrite');
    const store = tx.objectStore('embeddings');
    const request = store.clear();
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
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
      const existing = await getEmbedding(key);
      if (existing) {
        done++;
        onProgress?.(done, jobs.length);
        continue;
      }
    }
    try {
      const embedding = await embedImage(url);
      await setEmbedding(key, { code, url, embedding: Array.from(embedding) });
      indexed++;
    } catch (err) {
      errors.push({ url, error: err.message });
      console.warn('Не удалось проиндексировать:', url, err.message);
    }
    done++;
    onProgress?.(done, jobs.length);
  }

  return { total: jobs.length, indexed, errors };
}

export async function findSimilar(photo, topN = 5) {
  const queryEmbedding = await embedImage(photo);
  const all = await getAllEmbeddings();

  const scored = [];
  for (const item of all) {
    if (!item.id || !item.id.startsWith(DB_PREFIX)) continue;
    const embedding = Float32Array.from(item.embedding);
    scored.push({
      code: item.code,
      imageUrl: item.url,
      score: cosine(queryEmbedding, embedding),
    });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topN);
}

export async function getIndexSize() {
  const all = await getAllEmbeddings();
  return all.filter(item => item.id && item.id.startsWith(DB_PREFIX)).length;
}

export async function clearIndex() {
  await clearAll();
}