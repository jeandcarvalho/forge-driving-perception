// ============================
// SearchQuery — filtros de busca unificados
// Usado tanto no frontend (montar URL) quanto no backend (parsear query)
// ============================

export type SearchQuery = {
  page?: string;
  per_page?: string;

  // blocks
  "b.vehicle"?: string;
  "b.period"?: string;
  "b.condition"?: string;

  // CAN
  "c.v"?: string;
  "c.swa"?: string;
  "c.brakes"?: string;

  // Semseg
  "s.building"?: string;
  "s.vegetation"?: string;
  "s.sidewalk"?: string;

  // YOLO
  "y.class"?: string;
  "y.rel"?: string;
  "y.conf"?: string;
  "y.dist_m"?: string;
  "y.volume"?: string;

  // GPS spacing post-filter
  "gps_min_dist"?: string;
};

// ============================
// Search result types
// ============================

export type SearchHit = {
  acq_id: number | null;
  sec: number;
  link?: string;
};

export type SearchPageInfo = {
  page: number;
  per_page: number;
  has_more: boolean;
  total: number;
  total_pages: number;
};

export type SearchCounts = {
  matched_acq_ids: number;
  matched_seconds: number;
};

export type SearchResponse = {
  page: number;
  per_page: number;
  has_more: boolean;
  counts: SearchCounts;
  page_info: SearchPageInfo;
  matched_acq_ids: number;
  total_hits: number;
  items: SearchHit[];
};

// ============================
// Acquisition link types
// ============================

export type AcquisitionLink = {
  ext: string;
  link: string;
  sec: number | null;
};

// ============================
// Auth types
// ============================

export type AuthUser = {
  id: string;
  name: string | null;
  email: string;
  createdAt: string;
};

export type LoginRequest = {
  email: string;
  password: string;
};

export type RegisterRequest = {
  name?: string;
  email: string;
  password: string;
};

export type AuthResponse = {
  token: string;
  user: AuthUser;
};

// ============================
// Collection types
// ============================

export type Collection = {
  id: string;
  name: string;
  description: string | null;
  createdAt: string;
  _count?: { items: number };
};

export type CollectionItem = {
  acq_id: string;
  sec: number;
};

// ============================
// LLM types
// ============================

export type LLMTest = {
  testName: string;
  llmModel: string;
  promptType: string;
  count: number;
};

export type LLMResponse = {
  acq_id: string;
  centerSec: number;
  secs: number[];
  prompt: string | null;
  response: string | null;
  totalTokens: number | null;
  latencyMs: number | null;
};

export type LLMTestEval = {
  acq_id: string;
  sec: number;
  testName: string;
  llmModel: string;
  promptType: string;
  test1: number;
  test2: number;
  test3: number;
  test4: number;
  test5: number;
};
