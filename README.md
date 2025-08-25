

## Environment variables
 - `MONGO_URI` — full MongoDB connection string (preferred).
 - `MONGO_USER` and `MONGO_PASSWORD` — if you prefer to build the URI from parts (used with `MONGO_HOST`).
 - `MONGO_HOST` — host for Mongo (default: localhost:27017). For Atlas use the cluster host (e.g. cluster0.xyz.mongodb.net).
 - `MONGO_DB` — database name (default: Endeavor).
 - `MONGO_COLLECTION` — collection name (default: ragCollection).

Note: For CPU-heavy LLMs, consider using a dedicated worker service or Render cron/jobs to offload processing.

### Avoiding CUDA on Render
By default some wheels may pull CUDA-enabled PyTorch. To force a CPU-only install we pin a CPU wheel in `requirements.txt`.

If you already deployed and Render installed CUDA-enabled torch, redeploy after updating the repo (commit & push the changed `requirements.txt`). In Render: open your service -> Manual Deploy -> Deploy Latest Revision.

Render Python runtime requirement
---------------------------------
- Set the service runtime to **Python 3.11** in Render (Environment -> Runtime). PyTorch wheels in the official find-links are most compatible with Python 3.11 on Render.
- We include a `--find-links` entry in `requirements.txt` to pull CPU-only PyTorch wheels from PyTorch's stable CPU index.

Requirements files
------------------
- `requirements.txt` — slim production requirements for deployment (suitable for Render). Does NOT include heavy local ML stacks.
- `requirements-dev.txt` — development requirements that include `torch` and `sentence-transformers` for local embedding experiments. Install locally with:

```bash
pip install -r requirements-dev.txt
```

On Render use the normal `requirements.txt` to keep builds small.


