
# Applications


| Name | Main files | Compose name | URLs | Description |
|---|---|---|---|---|
| Support Bot | `bot.py` | `bot` | http://localhost:8501 | Main usecase. Fullstack Python application. |
| ESIMO  Loader | `loader.py` | `loader` | http://localhost:8502 | Load SO data into the database (create vector embeddings etc). Fullstack Python application. |
| PDF Reader | `pdf_bot.py` | `pdf_bot` | http://localhost:8503 | Read local PDF and ask it questions. Fullstack Python application. |
| Standalone Bot API | `api.py` | `api` | http://localhost:8504 | Standalone HTTP API streaming (SSE) + non-streaming endpoints Python. |
| Standalone Bot UI | `front-end/` | `front-end` | http://localhost:8505 | Standalone client that uses the Standalone Bot API to interact with the model. JavaScript (Svelte) front-end. |

The database can be explored at http://localhost:7474.
