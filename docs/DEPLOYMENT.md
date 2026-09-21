# Deployment

This repository holds two things with very different hosting needs.

| | What it is | Where it can run |
| --- | --- | --- |
| `index.html` | The cognitive games. One self-contained page, no backend. | **Vercel** |
| Everything else | Django + Channels voice assistant and screening models. | A host that runs a persistent process |

## The games, on Vercel

`vercel.json` and `.vercelignore` are set up so Vercel serves the games page
and nothing else.

```bash
npx vercel --prod
```

Or connect the repository in the Vercel dashboard — the config is picked up
automatically. Nothing else is needed: the page pulls Tailwind and Lucide from
a CDN and keeps its state in the browser.

## Why the assistant cannot go on Vercel

Not a configuration problem. Four things about the application are
incompatible with serverless functions:

1. **Websockets.** The voice assistant is a Channels application. A turn holds
   an open socket while audio streams in and speech streams back. Vercel
   functions are request/response and terminate when the response is sent.
2. **A background scheduler.** Reminders are delivered by a polling task that
   has to outlive any single request. There is no long-lived process to run it
   in, so a reminder set at 2pm for 3pm would never fire.
3. **Size.** torch, TensorFlow and transformers come to several gigabytes
   against a 250 MB unzipped limit for a function bundle.
4. **State on disk.** The FAISS store and SQLite database live on the
   filesystem. Serverless filesystems are ephemeral and not shared between
   invocations, so memories would disappear between turns.

The first two are architectural. Even with the models moved behind an API,
the websocket and the scheduler still need somewhere to live.

## Where the assistant should go

Anything that runs a container or a persistent process: **Render**,
**Railway**, **Fly.io**, or a plain VM. A `Dockerfile` is included.

```bash
docker build -t synapse .
docker run -p 8000:8000 --env-file .env synapse
```

The image ships without the inference wheels, because they are several
gigabytes and the app degrades cleanly without them — reminders, conversation
and the dashboard all work, and anything needing a model says so. To include
them:

```bash
docker build --build-arg WITH_ML=1 -t synapse .
```

### Required configuration

`python manage.py check --deploy` passes with these set. It fails loudly if
`SECRET_KEY` is too short, rather than starting with a weak one.

| Variable | Notes |
| --- | --- |
| `SECRET_KEY` | At least 50 characters. `python -c "from django.core.management.utils import get_random_secret_key as k; print(k())"` |
| `DEBUG` | `False`. Turning it off enables HTTPS redirect, HSTS, secure cookies and `X-Frame-Options: DENY`. |
| `ALLOWED_HOSTS` | Your domain. Also derives `CSRF_TRUSTED_ORIGINS`. |
| `TIME_ZONE` | Where the person is. Reminders are spoken in local terms. |
| `OLLAMA_URL` | The intent router. Must be reachable from the container. |
| `MISTRAL_API_KEY` | The reasoning layer. |

Static files are served by WhiteNoise, so no separate web server is needed.
TLS is assumed to terminate at the platform's proxy; `SECURE_PROXY_SSL_HEADER`
is set for `X-Forwarded-Proto`.

### The router needs somewhere to run too

Ollama is a separate service. Either run it as a second container on the same
private network and point `OLLAMA_URL` at it, or swap the router to a hosted
API. `qwen2.5:1.5b-instruct` needs roughly 1.2 GB of VRAM.

### Before going live

- **Rotate the Gemini key** in commit `ee9011f`. It is in the public history.
- Move off SQLite if more than a handful of people will use it.
- Put the FAISS store on a persistent volume, or it resets on every deploy.
- Re-read the audio indicator's limits in the README. It reaches a conclusion
  for about a third of recordings, and is not a diagnostic tool.
