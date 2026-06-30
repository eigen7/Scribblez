"""Training dashboard: SQLite data store + Tornado data API for the React app.

`db` is the per-tag SQLite store; `plots` builds the Bokeh figures; `api` serves
them (as embeddable json_items) and the lane-analysis data over HTTP; `react_server`
launches the React app + the API. See docs/react_dashboard.md.
"""
