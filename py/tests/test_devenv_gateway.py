"""Tests for the gateway service (submodules/devenv_utils/gateway_service.py) and
the [services] config parsing that feeds it.

Pure config parsing, URL/label construction, and the env-var contract -- no
Traefik container, no Docker, no ~/.devenv.
"""

import sys
from pathlib import Path

import pytest

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scribblez.service_urls import service_url  # noqa: E402

from submodules.devenv_utils import gateway_service  # noqa: E402
from submodules.devenv_utils.config import DevenvConfig, Service, load_config  # noqa: E402
from submodules.devenv_utils.console import SetupException  # noqa: E402

# ---- [services] parsing --------------------------------------------------


def _write_toml(tmp_path: Path, body: str) -> Path:
    (tmp_path / "devenv.toml").write_text(body)
    return tmp_path


def test_services_int_form_parses_to_service(tmp_path: Path):
    cfg = load_config(_write_toml(tmp_path, 'name = "proj"\n[services]\nweb = 5173\n'))
    assert cfg.services == {"web": Service(port=5173, publish=False)}


def test_services_table_form_with_publish(tmp_path: Path):
    cfg = load_config(
        _write_toml(
            tmp_path,
            'name = "proj"\n[services]\nweb = 5173\ndebugger = { port = 9229, publish = true }\n',
        )
    )
    assert cfg.services["web"] == Service(port=5173, publish=False)
    assert cfg.services["debugger"] == Service(port=9229, publish=True)


def test_invalid_service_name_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="service name"):
        load_config(_write_toml(tmp_path, 'name = "proj"\n[services]\nWeb = 5173\n'))


def test_invalid_project_name_rejected_when_services_present(tmp_path: Path):
    with pytest.raises(ValueError, match="project name"):
        load_config(_write_toml(tmp_path, 'name = "a_b"\n[services]\nweb = 5173\n'))


def test_service_round_trips_through_direct_config(tmp_path: Path):
    cfg = DevenvConfig(
        name="proj", repo_root=tmp_path, services={"web": Service(port=3000, publish=True)}
    )
    assert cfg.services == {"web": Service(port=3000, publish=True)}


# ---- Pure URL / label construction ---------------------------------------


def test_service_hostname():
    assert gateway_service.service_hostname("scribblez", "web") == "scribblez-web.localhost"


def test_service_env_var_maps_hyphen_to_underscore():
    assert gateway_service.service_env_var("my-svc") == "DEVENV_SERVICE_URL_MY_SVC"


def test_service_url_default_port_has_no_suffix():
    assert gateway_service.service_url("scribblez", "web", 80) == "http://scribblez-web.localhost"


def test_service_url_non_default_port_carries_suffix():
    assert (
        gateway_service.service_url("scribblez", "web", 8080)
        == "http://scribblez-web.localhost:8080"
    )


def test_router_labels_exact_list():
    assert gateway_service.router_labels("scribblez", "web", 5173) == [
        "--label", "traefik.enable=true",
        "--label", "traefik.http.routers.scribblez-web.rule=Host(`scribblez-web.localhost`)",
        "--label", "traefik.http.routers.scribblez-web.entrypoints=web",
        "--label", "traefik.http.routers.scribblez-web.service=scribblez-web",
        "--label", "traefik.http.services.scribblez-web.loadbalancer.server.port=5173",
    ]  # fmt: skip


def test_container_args_includes_labels_env_and_publish():
    services = {
        "web": Service(port=5173, publish=False),
        "debugger": Service(port=9229, publish=True),
    }
    args = gateway_service.container_args("scribblez", services, 80)

    # Routing labels for both services, with traefik.enable emitted exactly once.
    assert args.count("traefik.enable=true") == 1
    assert "traefik.http.services.scribblez-web.loadbalancer.server.port=5173" in args
    debugger_host = gateway_service.service_hostname("scribblez", "debugger")
    assert f"traefik.http.routers.scribblez-debugger.rule=Host(`{debugger_host}`)" in args

    # A DEVENV_SERVICE_URL_* env var per service.
    assert "DEVENV_SERVICE_URL_WEB=http://scribblez-web.localhost" in args
    assert "DEVENV_SERVICE_URL_DEBUGGER=http://scribblez-debugger.localhost" in args

    # -p only for the publish=true service.
    assert "127.0.0.1:9229:9229" in args
    assert "127.0.0.1:5173:5173" not in args


# ---- Host-network + launch URL wiring (no docker, no ~/.devenv) -----------


def _config_with_services(services: dict) -> DevenvConfig:
    return DevenvConfig(name="scribblez", repo_root=Path("/tmp"), services=services)


def test_host_network_urls_use_plain_localhost():
    services = {"web": Service(port=5173), "dash": Service(port=5180)}
    assert gateway_service.host_network_urls(services) == {
        "web": "http://localhost:5173",
        "dash": "http://localhost:5180",
    }


def test_launch_urls_empty_without_services():
    cfg = _config_with_services({})
    assert gateway_service.launch_urls(cfg, host_network=False) == {}


def test_launch_urls_host_network_bypasses_gateway():
    cfg = _config_with_services({"web": Service(port=5173)})
    assert gateway_service.launch_urls(cfg, host_network=True) == {"web": "http://localhost:5173"}


def test_launch_urls_gateway_mode_uses_configured_port(monkeypatch):
    monkeypatch.setattr(gateway_service, "load_service_config", lambda: {"http_port": 80})
    cfg = _config_with_services({"web": Service(port=5173)})
    assert gateway_service.launch_urls(cfg, host_network=False) == {
        "web": "http://scribblez-web.localhost"
    }


def test_launch_urls_gateway_mode_raises_when_unprovisioned(monkeypatch):
    monkeypatch.setattr(gateway_service, "load_service_config", lambda: None)
    cfg = _config_with_services({"web": Service(port=5173)})
    with pytest.raises(SetupException, match="gateway service has not been set up"):
        gateway_service.launch_urls(cfg, host_network=False)


# ---- scribblez.service_urls (in-container URL contract) -------------------


def test_service_url_returns_env_when_port_matches_default(monkeypatch):
    monkeypatch.setenv("DEVENV_SERVICE_URL_DASH", "http://scribblez-dash.localhost")
    assert service_url("dash", 5180, 5180) == "http://scribblez-dash.localhost"


def test_service_url_falls_back_when_port_overridden(monkeypatch):
    monkeypatch.setenv("DEVENV_SERVICE_URL_DASH", "http://scribblez-dash.localhost")
    # An overridden port has no gateway route, so the localhost form wins.
    assert service_url("dash", 5999, 5180) == "http://localhost:5999"


def test_service_url_falls_back_when_env_unset(monkeypatch):
    monkeypatch.delenv("DEVENV_SERVICE_URL_DASH", raising=False)
    assert service_url("dash", 5180, 5180) == "http://localhost:5180"
