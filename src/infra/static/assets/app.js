/* 演示前端共用脚本（零构建，原生 fetch）。
 *
 * 这里只放「两个页面都会用到」的东西，业务逻辑留给各自的 <script>：
 * - 请求封装：统一带凭证、统一解析后端错误信封（后端所有错误都是 {success:false,error:{...}}）；
 * - 凭证存储：会员 token（Authorization: Bearer）与项目 API Key（X-API-Key）分开放，
 *   因为后端本来就把「买家面」与「运营面」设计成互不通用的两套凭证；
 * - 展示小工具：金额、时间、HTML 转义、JSON 折叠块、toast。
 *
 * 诚实边界：本文件不做任何核心信息泄露之外的「安全」承诺 ——
 * 演示页把 API Key 放在浏览器里只是为了免去搭一个后端的麻烦，
 * 真实项目里运营台凭证必须由服务端会话持有（见 PROJECT.md 的诚实边界）。
 */

const Store = {
  memberToken: {
    get: () => localStorage.getItem("lumiere.member_token") || "",
    set: (v) => (v ? localStorage.setItem("lumiere.member_token", v) : localStorage.removeItem("lumiere.member_token")),
  },
  adminKey: {
    get: () => localStorage.getItem("lumiere.admin_key") || "",
    set: (v) => (v ? localStorage.setItem("lumiere.admin_key", v) : localStorage.removeItem("lumiere.admin_key")),
  },
  member: {
    get: () => {
      try { return JSON.parse(localStorage.getItem("lumiere.member") || "null"); } catch { return null; }
    },
    set: (v) => (v ? localStorage.setItem("lumiere.member", JSON.stringify(v)) : localStorage.removeItem("lumiere.member")),
  },
};

/** 一次后端调用的统一入口。
 *  - key: 项目级 API Key（写 X-API-Key）
 *  - token: 会员 token（写 Authorization: Bearer）
 *  返回解析后的 JSON；非 2xx 时抛出 Error，并挂上 .status / .envelope，方便调用方区分 401/403/429。 */
async function api(path, { method = "GET", body, key, token, raw = false } = {}) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const apiKey = key ?? Store.adminKey.get();
  const memberToken = token ?? Store.memberToken.get();
  if (apiKey) headers["X-API-Key"] = apiKey;
  if (memberToken) headers["Authorization"] = `Bearer ${memberToken}`;

  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (raw) return res;

  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }

  if (!res.ok) {
    const err = new Error((data && data.error && data.error.message) || `请求失败 (${res.status})`);
    err.status = res.status;
    err.envelope = data;
    throw err;
  }
  return data;
}

/** 解析后端错误信封，给出一句可读的提示（含 error code 与 trace_id）。 */
function describeError(err) {
  const env = err && err.envelope;
  if (env && env.error) {
    const code = env.error.code ? `[${env.error.code}] ` : "";
    return `${code}${env.error.message || err.message}`;
  }
  return (err && err.message) || "未知错误";
}

const fmt = {
  usd: (n) => `$${(Number(n) || 0).toFixed(2)}`,
  usd6: (n) => `$${(Number(n) || 0).toFixed(6)}`,
  cents: (c) => fmt.usd((Number(c) || 0) / 100),
  num: (n) => (Number(n) || 0).toLocaleString("en-US"),
  time: (iso) => {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    const p = (x) => String(x).padStart(2, "0");
    return `${d.getMonth() + 1}/${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  },
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** 折叠展示一段后端原始响应：演示页要让「后端真的返回了什么」可见。 */
function jsonBlock(title, obj) {
  return `<details class="ai-trace"><summary>${esc(title)}</summary><pre class="json">${esc(JSON.stringify(obj, null, 2))}</pre></details>`;
}

function toast(message, kind = "") {
  let wrap = document.querySelector(".toast-wrap");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "toast-wrap";
    document.body.appendChild(wrap);
  }
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  wrap.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

/** 把表单里的「空字符串」清成 undefined：后端很多 Query 参数留空即「不限」，传空串会被拒。 */
function compact(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v !== "" && v !== null && v !== undefined) out[k] = v;
  }
  return out;
}

/** 图片：全部走平台的文生图接口（演示页禁止用占位图）。 */
function imgUrl(prompt, size = "square") {
  return `https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=${encodeURIComponent(prompt)}&image_size=${size}`;
}

/** 按品类选一张稳定的美妆配图（同一商品每次渲染拿到同一张）。 */
const CATEGORY_PROMPT = {
  "精华": "luxury serum glass dropper bottle on marble podium, soft pink light, minimal beauty product photography",
  "面霜": "premium face cream jar with rose petals, cream white background, editorial beauty still life",
  "洁面": "gentle facial cleanser tube on wet stone, soft morning light, clean skincare product photo",
  "口红": "luxury lipstick bullet on silk fabric, rose gold accents, high end cosmetic advertisement",
  "香水": "elegant perfume flacon with gold cap, soft neutral backdrop, fragrance advertisement photography",
  "面膜": "hydrating sheet mask pack with water droplets, fresh green leaves, skincare product photography",
};
function productImage(product) {
  const prompt = CATEGORY_PROMPT[product.category] || `high end ${product.category || "beauty"} cosmetic product, luxury brand, minimal studio light`;
  return imgUrl(`${prompt}`, "square");
}

function statusBadge(item) {
  const map = { ON_SALE: ["ok", "在售"], OFF_SHELF: ["warn", "已下架"], DRAFT: ["", "草稿"] };
  const status = item.status || "";
  const [cls, label] = map[status] || ["", status || "—"];
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}

function orderBadge(status) {
  const map = {
    PENDING_PAYMENT: ["warn", "待付款"], PAID: ["brand", "已付款"],
    SHIPPED: ["ok", "已发货"], COMPLETED: ["ok", "已完成"], CANCELLED: ["err", "已取消"],
  };
  const [cls, label] = map[status] || ["", status || "—"];
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}
