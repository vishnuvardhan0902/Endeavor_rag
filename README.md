

## Environment variables
- `GOOGLE_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` (path) — required for evaluation.
- `WORKER_COUNT` — concurrency limit per process (default 6).
 - `MONGO_URI` — full MongoDB connection string (preferred).
 - `MONGO_USER` and `MONGO_PASSWORD` — if you prefer to build the URI from parts (used with `MONGO_HOST`).
 - `MONGO_HOST` — host for Mongo (default: localhost:27017). For Atlas use the cluster host (e.g. cluster0.xyz.mongodb.net).
 - `MONGO_DB` — database name (default: Endeavor).
 - `MONGO_COLLECTION` — collection name (default: ragCollection).


