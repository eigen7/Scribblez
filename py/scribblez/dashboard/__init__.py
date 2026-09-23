"""The master dashboard's Python backend: the Tornado API behind the React app
in web/, and the control plane that runs every workload's workers.

Data plane: `db` is the per-tag SQLite store; `plots` and
`worker_stats_figures` build Bokeh figures from it, which `api` serves as
json_items for the React app to embed (`figure_delta` streams their appended
rows); `api` and `trajectories_api` also serve the per-position analysis tabs.

Control plane: `tasks` holds the durable task records (task.json), `workers`
reconciles them with real processes, containers and rented machines, and
`master_api` exposes both to the web forms. `react_server` launches the API
and the Vite dev server. See docs/master_dashboard.md and
docs/react_dashboard.md.
"""
