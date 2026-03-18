"""Static testing-lab contract checks for dashboard, voice, and vision pages."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_suite_css_imports_workflow_layer() -> None:
    """The shared suite stack should include the workflow board styling layer."""
    css = _read("static/assets/css/suite.css")
    assert '@import "./suite/workflows.css";' in css


def test_dashboard_includes_connected_route_boards() -> None:
    """The dashboard should expose scenario and route boards plus workflow scripts."""
    html = _read("static/index.html")
    assert 'id="dashboardScenarioCatalog"' in html
    assert 'id="dashboardRouteBoard"' in html
    assert "./assets/css/suite/critical-shell.css?v=20260318-shellfix" in html
    assert "./assets/css/suite.css?v=20260318-shellfix" in html
    assert "./assets/js/agent-workflow-data.js" in html
    assert "./assets/js/agent-workflows.js" in html
    assert html.index("./assets/js/agent-workflow-data.js") < html.index("./assets/js/agent-workflows.js")
    assert html.index("./assets/js/agent-workflows.js") < html.index("./assets/js/dashboard.js")


def test_voice_hub_includes_rest_and_duplex_workflow_boards() -> None:
    """Voice Hub should render explicit route and duplex contract boards."""
    html = _read("static/voice_agent.html")
    assert 'id="voiceScenarioCatalog"' in html
    assert 'id="restWorkflowBoard"' in html
    assert 'id="wsWorkflowBoard"' in html
    assert "./assets/css/suite/critical-shell.css?v=20260318-shellfix" in html
    assert "./assets/css/suite.css?v=20260318-shellfix" in html
    assert "./assets/js/lab-state.js" in html
    assert "./assets/js/agent-workflow-data.js" in html
    assert "./assets/js/agent-workflows.js" in html
    assert "./assets/js/voice-hub-lab.js" in html
    assert html.index("./assets/js/lab-state.js") < html.index("./assets/js/agent-workflow-data.js")
    assert html.index("./assets/js/agent-workflows.js") < html.index("./assets/js/voice-hub-lab.js")
    assert html.index("./assets/js/voice-hub-lab.js") < html.index("./assets/js/voice-agent-rest.js")


def test_voice_hub_scripts_reference_accurate_route_and_health_fields() -> None:
    """The JS should wire the new workflow boards and read current health payload fields."""
    rest_js = _read("static/assets/js/voice-agent-rest.js")
    ws_js = _read("static/assets/js/voice-agent-ws.js")
    tools_js = _read("static/assets/js/voice-agent-tools.js")
    assert 'renderVoiceWorkflow("restWorkflowBoard", data)' in rest_js
    assert "saveVoiceHandoff(data)" in rest_js
    assert "VoiceHubLab?.handleWsMessage(msg)" in ws_js
    assert "stt_providers" in tools_js
    assert "tts_provider" in tools_js


def test_vision_lab_page_wires_real_assess_and_attach_flow() -> None:
    """Vision Lab should connect the shared vision API and listing grade route."""
    html = _read("static/vision_lab.html")
    js = _read("static/assets/js/vision-lab.js")
    assert 'id="visionScenarioCatalog"' in html
    assert 'id="visionPipelineBoard"' in html
    assert "./assets/css/suite/critical-shell.css?v=20260318-shellfix" in html
    assert "./assets/css/suite.css?v=20260318-shellfix" in html
    assert "./assets/js/lab-state.js" in html
    assert "./assets/js/agent-workflow-data.js" in html
    assert "./assets/js/agent-workflows.js" in html
    assert "./assets/js/vision-lab.js" in html
    assert "/api/v1/vision/health" in js
    assert "/api/v1/vision/assess" in js
    assert "/api/v1/listings/" in js


def test_all_suite_pages_link_versioned_shell_assets() -> None:
    """Suite and premium pages should link critical shell CSS directly to avoid stale bundle issues."""
    suite_pages = [
        "static/index.html",
        "static/voice_agent.html",
        "static/vision_lab.html",
        "static/rag_test.html",
        "static/voice_realtime.html",
        "static/voice_test_ui.html",
    ]
    for relative_path in suite_pages:
        html = _read(relative_path)
        assert "./assets/css/suite/critical-shell.css?v=20260318-shellfix" in html
        assert "./assets/css/suite.css?v=20260318-shellfix" in html
    premium_html = _read("static/premium_voice.html")
    assert "./assets/css/premium-voice/critical-rail.css?v=20260318-shellfix" in premium_html
    assert "./assets/css/premium-voice.css?v=20260318-shellfix" in premium_html


def test_new_static_workflow_files_stay_within_small_file_limit() -> None:
    """New workflow-focused assets should stay within the repo's small-file rule."""
    targets = [
        "static/assets/css/suite/workflows.css",
        "static/assets/css/suite/critical-shell.css",
        "static/assets/css/premium-voice/critical-rail.css",
        "static/assets/css/vision-lab.css",
        "static/assets/js/agent-workflow-data.js",
        "static/assets/js/agent-workflows.js",
        "static/assets/js/lab-state.js",
        "static/assets/js/vision-lab.js",
        "static/assets/js/voice-hub-lab.js",
        "static/assets/js/voice-agent-rest.js",
        "tests/unit/test_static_testing_lab.py",
    ]
    for relative_path in targets:
        line_count = len(_read(relative_path).splitlines())
        assert line_count <= 200, f"{relative_path} exceeded 200 lines with {line_count}"
