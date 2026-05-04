# VideoAmp API Endpoints

Base URL: `https://api.videoamp.dev`

Time estimates assume 2 workers at ~33 records/min.

---

## Content

### list-episodes
- **GET** `/external/v1/content/episodes`
- **Dir:** `episodes/`
- **Params:** pageSize, pageToken, networkId, programId, currencyOfRecord
- Returns paginated list of TV episodes with metadata.
- **Sweep:** 8/8 variants confirmed · 240 records · ~7 min

### list-programs
- **GET** `/external/v1/content/programs`
- **Dir:** `programs/`
- **Params:** pageSize, pageToken, networkId, currencyOfRecord, name (fuzzy), programIds (array)
- Returns programs (shows) available for content measurement.
- **Sweep:** 8/8 variants confirmed · 240 records · ~7 min

### list-networks
- **GET** `/external/v1/content/networks`
- **Dir:** `networks/`
- **Params:** pageSize, pageToken, currencyOfRecord, name (fuzzy), networkIds (array), mediaGroupName
- Returns networks (broadcasters/cable channels) available for filtering.
- **Sweep:** 4/4 variants confirmed · 120 records · ~4 min

### list-network-mediagroups
- **GET** `/external/v1/content/media-groups`
- **Dir:** `media-groups/`
- **Params:** pageSize, pageToken, name (fuzzy)
- Returns media groups — parent organizations that own multiple networks.
- **Sweep:** 2/2 variants confirmed · 60 records · ~2 min

### list-metric-and-dimension-types
- **GET** `/external/v1/content/metric-type-compatibility-matrix`
- **Dir:** `metric-types/`
- **Params:** none
- Returns the full compatibility matrix of supported metrics and dimensions for content measurement. No filtering — always returns the full matrix.
- **Sweep:** 1/1 variants confirmed · 30 records · ~1 min

### get-episode
- **GET** `/external/v1/content/episodes/{episodeId}`
- **Dir:** `episode/`
- **Params:** episodeId (integer path param)
- Returns detailed metadata for a single episode by ID.
- **Sweep:** 1/1 variants confirmed · 30 records · ~1 min (IDs pre-seeded — sequential from 1, thousands valid)

### get-program
- **GET** `/external/v1/content/programs/{programId}`
- **Dir:** `program/`
- **Params:** programId (integer path param)
- Returns details for a single program by ID.
- **Sweep:** 1/1 variants confirmed · 30 records · ~1 min (IDs pre-seeded — sequential from 1, thousands valid)

### get-network
- **GET** `/external/v1/content/networks/{id}`
- **Dir:** `network/`
- **Params:** id (integer path param)
- Returns details for a single network by ID.
- **Sweep:** 1/1 variants confirmed · 30 records · ~1 min (22 valid IDs seeded from episodes sweep)

### get-metrics
- **GET** `/external/v1/content/metrics/{id}`
- **Dir:** `content-metric/`
- **Params:** id (UUID string path param)
- Returns status and output location for a content metrics job. UUIDs come from `POST /external/v1/content/metrics` — cannot be swept.
- **Sweep:** Blocked — account lacks permission to POST `/external/v1/content/metrics` (403 on create-metrics). No UUID obtainable via this account.

---

## Measurements

### list-ad
- **GET** `/external/v1/measurements`
- **Dir:** `measurements/`
- **Params:** pageSize, pageToken, currencyOfRecord, status (failed/processing/ready), name (fuzzy), createdAt (YYYY-MM-DD), orderBy
- Returns paginated list of ad measurement requests with status and metadata.
- **Sweep:** 4/8 variants confirmed · 120 records · ~4 min (pageToken variants fail — test token is endpoint-specific)

### get-ad
- **GET** `/external/v1/measurements/{id}`
- **Dir:** `measurement/`
- **Params:** id (UUID path param — requestId)
- Returns status, results, and output file location for a single ad measurement request.
- **Sweep:** 1/1 variants confirmed · 30 records · ~1 min (UUID seeded)

---

## Audiences

### list
- **GET** `/v1/audiences`
- **Dir:** `audiences/`
- **Params:** pageSize, pageToken, status, type (advanced/demo/shared/owned/linear_commercial/exposure/content), level (HOUSEHOLD/PERSON), orderBy, query (fuzzy), useCases (measurement/activation), year (broadcast year 2020–2099)
- Returns paginated list of audiences accessible to the user. Richest filter set of any endpoint.
- **Sweep:** 8/8 variants confirmed · 240 records · ~7 min

### get
- **GET** `/v1/audiences/{id}`
- **Dir:** `audience/`
- **Params:** id (UUID or integer path param)
- Returns details for a single audience. Accepts both UUID and legacy integer ID.
- **Sweep:** 1/1 variants confirmed · 30 records · ~1 min (ID 232962 seeded)

### list-statuses
- **GET** `/v1/audiences/status`
- **Dir:** `audience-statuses/`
- **Params:** pageSize, pageToken, status (ready/failed/processing/draft), orderBy, requestId (UUID)
- Returns audience creation statuses. Useful for polling bulk creation jobs.
- **Sweep:** 4/4 variants confirmed · 120 records · ~4 min

### list-export
- **GET** `/v1/audiences/{audienceId}/exports`
- **Dir:** `audience-exports/`
- **Params:** audienceId (UUID or integer — required path param), pageSize, pageToken, sorts
- Returns exports associated with a specific audience. Requires a valid audienceId first.
- **Sweep:** 1/1 variants confirmed · 30 records · ~1 min (note: nested list endpoint — sorts/pageToken variants not yet tested)

---

## Top-Level

### currency-of-record
- **GET** `/external/v1/currency-of-record`
- **Dir:** `currency-of-record/`
- **Params:** pageSize, pageToken, currencyOfRecord (array filter), name (fuzzy), reportingScope (exact, case-insensitive)
- Returns available Currency of Record methodologies with broadcast year context.
- **Sweep:** 2/2 variants confirmed · 60 records · ~2 min

### me
- **GET** `/v1/me`
- **Dir:** `me/`
- **Params:** none
- Returns details about the currently authenticated user. Training data is phrasing variety only.
- **Sweep:** 1/1 variants confirmed · 30 records · ~1 min

### consents
- **GET** `/v1/consents`
- **Dir:** `consents/`
- **Params:** pageSize, pageToken, fetchRecipientAncestorPath (boolean, default true), q (structured filter: recipient_kind EQ/IN, recipient_name STARTSWITH/ENDSWITH/CONTAINS)
- Returns consents between your organization and consenting recipients.
- **Sweep:** 2/2 variants confirmed · 60 records · ~2 min
