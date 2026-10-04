# Pull protocol v1

How a site hands image-to-mesh jobs to `img2mesh serve`. The worker only makes
**outbound** HTTPS requests, so it can run behind a home router with no open
ports, and it never holds database or storage keys: the site gives it
short-lived signed URLs instead.

## Auth

Every request to the site carries:

```
Authorization: Bearer <WORKER_SECRET>
Content-Type: application/json
```

and every JSON body includes `"protocol": 1` and `"worker": "<WORKER_NAME>"`.
The site rejects a wrong secret with `401` and an unknown protocol with `400`.

## 1. Claim a job

`POST {SITE_URL}/api/worker/claim` with `{"protocol": 1, "worker": "pc-1"}`

- `204 No Content`: nothing to do. The worker waits `POLL_SECONDS` (default 5) and asks again.
- `200`:

```json
{
  "job": {
    "id": "c0ffee…",
    "image_url": "https://…signed GET URL for the input image…",
    "upload_url": "https://…signed PUT URL for the result STL…",
    "options": { "height_mm": 150 }
  }
}
```

The site must hand each job to one worker only (for example
`select … for update skip locked`) and mark it running with the worker's name.

### `options`

| Key | Default | Meaning |
|---|---|---|
| `height_mm` | **required** | Final height of the print, in millimetres |
| `remove_background` | `"auto"` | `auto` (only if the image has no transparency), `always`, `never` |
| `flat_back` | `0.04` | Share of the model's depth cut off the back, 0–0.5 |
| `flat_bottom` | `0.02` | Share of the model's height cut off the bottom, 0–0.5 |
| `seed` | `1234` | Change for a different result from the same image |
| `steps` | `50` | Diffusion steps, 1–200 |
| `octree_resolution` | `384` | Mesh detail: 256, 384 or 512 |
| `up`, `front` | `"+y"`, `"+z"` | Axes of the raw Hunyuan output; rarely changed |

Invalid options fail the job (step 3), not the worker.

## 2. While running: heartbeat

Every 15 seconds:

`POST {SITE_URL}/api/worker/jobs/{id}` with
`{"protocol": 1, "worker": "pc-1", "event": "heartbeat", "progress": 0.42}`

`progress` is 0–1 or `null`. A site should put a running job whose last
heartbeat is a few minutes old back in the queue: its worker died.

## 3. Finish

On success the worker first uploads the binary STL with
`PUT {upload_url}` (`Content-Type: model/stl`), then reports:

```json
{"protocol": 1, "worker": "pc-1", "event": "succeeded",
 "stats": {"seconds": {"background": 1.2, "shape": 48.0, "cleanup": 6.1, "printable": 0.8},
           "faces": 298112, "watertight": true, "size_mm": [118.4, 31.0, 150.0]}}
```

On failure:

```json
{"protocol": 1, "worker": "pc-1", "event": "failed", "error": "RuntimeError: CUDA out of memory"}
```

Both go to `POST {SITE_URL}/api/worker/jobs/{id}`. Answer `200` to all three
events. The STL is in millimetres, Z up, standing on Z = 0, front facing −Y.

## Versioning

A breaking change becomes protocol 2, and the worker sends `"protocol": 2`.
A site may support both while workers are upgraded.
