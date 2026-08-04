"""Tests for the failure/progress notification channels.

The property that matters most here is negative: a notifier exists to report
that something else broke, so it must not itself be able to break a run. Every
test that exercises a failure path asserts that the failure was swallowed.
"""

from __future__ import annotations

import urllib.error

import pytest

from method import notify
from method.notify import Heartbeat, Notifier


class _FakeResponse:
    """Stands in for the context manager ``urlopen`` returns."""

    def __init__(self, _ignored=None) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def read(self) -> bytes:
        return b"OK"


@pytest.fixture(autouse=True)
def no_notify_env(monkeypatch):
    """Start every test from an unconfigured environment.

    Otherwise a developer's own .env (already loaded in some other test) would
    decide whether these pass, and a real email would be a possible outcome of
    running the suite.
    """
    for var in (
        notify.RESEND_API_KEY_ENV,
        notify.NOTIFY_EMAIL_ENV,
        notify.NOTIFY_FROM_ENV,
        notify.NOTIFY_TAG_ENV,
        notify.HEARTBEAT_URL_ENV,
    ):
        monkeypatch.delenv(var, raising=False)


class TestNotifierConfiguration:
    def test_unconfigured_is_disabled_and_sends_nothing(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("a disabled notifier must not touch the network")

        monkeypatch.setattr(notify, "_post_json", explode)
        notifier = Notifier.from_env()

        assert not notifier.enabled
        assert notifier.send("subject", "body") is False

    def test_key_without_recipient_is_disabled(self, monkeypatch):
        """A key with nobody to mail is as unusable as no key at all."""
        monkeypatch.setenv(notify.RESEND_API_KEY_ENV, "re_test")
        assert not Notifier.from_env().enabled

    def test_recipient_without_key_is_disabled(self, monkeypatch):
        monkeypatch.setenv(notify.NOTIFY_EMAIL_ENV, "me@example.com")
        assert not Notifier.from_env().enabled

    def test_recipients_are_comma_separated(self, monkeypatch):
        monkeypatch.setenv(notify.RESEND_API_KEY_ENV, "re_test")
        monkeypatch.setenv(notify.NOTIFY_EMAIL_ENV, "a@example.com, b@example.com")

        assert Notifier.from_env().recipients == ["a@example.com", "b@example.com"]

    def test_describe_never_leaks_the_key(self, monkeypatch):
        monkeypatch.setenv(notify.RESEND_API_KEY_ENV, "re_supersecret")
        monkeypatch.setenv(notify.NOTIFY_EMAIL_ENV, "me@example.com")

        assert "re_supersecret" not in Notifier.from_env().describe()

    def test_describe_names_what_is_missing(self):
        assert notify.RESEND_API_KEY_ENV in Notifier.from_env().describe()


class TestNotifierSending:
    def test_posts_subject_body_and_recipients(self, monkeypatch):
        captured = {}

        def capture(url, payload, headers):
            captured.update(url=url, payload=payload, headers=headers)

        monkeypatch.setattr(notify, "_post_json", capture)
        notifier = Notifier(
            api_key="re_test", recipients=["me@example.com"], sender="bot@example.com"
        )

        assert notifier.send("it broke", "traceback here") is True
        assert captured["payload"]["to"] == ["me@example.com"]
        assert captured["payload"]["from"] == "bot@example.com"
        assert captured["payload"]["subject"] == "it broke"
        assert captured["payload"]["text"] == "traceback here"
        assert captured["headers"]["Authorization"] == "Bearer re_test"

    def test_sends_an_explicit_user_agent(self, monkeypatch):
        """Cloudflare fronts the Resend API and bans urllib's default agent,
        answering 403 / error 1010 in a way that reads like a bad API key.
        Without an explicit User-Agent, nothing is ever delivered."""
        captured = {}
        monkeypatch.setattr(
            notify.urllib.request,
            "urlopen",
            lambda request, timeout: _FakeResponse(captured.update(request.headers)),
        )

        Notifier(api_key="k", recipients=["me@example.com"]).send("s", "b")

        agent = {key.lower(): value for key, value in captured.items()}["user-agent"]
        assert agent == notify._USER_AGENT
        assert not agent.startswith("Python-urllib")

    def test_heartbeat_ping_also_sets_the_user_agent(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            notify.urllib.request,
            "urlopen",
            lambda request, timeout: _FakeResponse(captured.update(request.headers)),
        )

        Heartbeat("https://hc.example/uuid").ping()

        assert {k.lower(): v for k, v in captured.items()}["user-agent"] == (
            notify._USER_AGENT
        )

    def test_tag_prefixes_the_subject(self, monkeypatch):
        """Two boxes working one family otherwise send indistinguishable mail."""
        captured = {}
        monkeypatch.setattr(
            notify, "_post_json", lambda url, payload, headers: captured.update(payload)
        )
        notifier = Notifier(api_key="k", recipients=["me@example.com"], tag="vast-4090")

        notifier.send("EXP3 done", "body")

        assert captured["subject"] == "[vast-4090] EXP3 done"

    def test_delivery_failure_is_swallowed_and_reported(self, monkeypatch):
        """The whole point: a dead notifier must not kill the run."""
        attempts = []

        def always_fail(url, payload, headers):
            attempts.append(url)
            raise urllib.error.URLError("network is down")

        monkeypatch.setattr(notify, "_post_json", always_fail)
        monkeypatch.setattr(notify.time, "sleep", lambda _seconds: None)
        notifier = Notifier(api_key="k", recipients=["me@example.com"])

        assert notifier.send("subject", "body") is False
        assert len(attempts) == notify._ATTEMPTS

    def test_transient_failure_is_retried_then_succeeds(self, monkeypatch):
        calls = []

        def fail_once(url, payload, headers):
            calls.append(url)
            if len(calls) == 1:
                raise urllib.error.URLError("blip")

        monkeypatch.setattr(notify, "_post_json", fail_once)
        monkeypatch.setattr(notify.time, "sleep", lambda _seconds: None)
        notifier = Notifier(api_key="k", recipients=["me@example.com"])

        assert notifier.send("subject", "body") is True
        assert len(calls) == 2


class TestHeartbeat:
    def test_disabled_heartbeat_is_an_inert_context_manager(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("a disabled heartbeat must not touch the network")

        monkeypatch.setattr(notify.urllib.request, "urlopen", explode)

        with Heartbeat.from_env() as beat:
            assert not beat.enabled

    def test_success_pings_start_then_bare_url(self, monkeypatch):
        pings = []
        monkeypatch.setattr(
            Heartbeat, "ping", lambda self, suffix="": pings.append(suffix)
        )

        with Heartbeat("https://hc.example/uuid", period=3600):
            pass

        assert pings == ["/start", ""]

    def test_failure_pings_fail(self, monkeypatch):
        """The fast path for a failure whose network still works; the watchdog
        timeout remains the backstop for one whose network does not."""
        pings = []
        monkeypatch.setattr(
            Heartbeat, "ping", lambda self, suffix="": pings.append(suffix)
        )

        with pytest.raises(RuntimeError):
            with Heartbeat("https://hc.example/uuid", period=3600):
                raise RuntimeError("CUDA out of memory")

        assert pings == ["/start", "/fail"]

    def test_trailing_slash_does_not_double_up(self):
        assert Heartbeat("https://hc.example/uuid/").url == "https://hc.example/uuid"

    def test_ping_swallows_network_errors(self, monkeypatch):
        """A missed ping is the signal, not an error to propagate."""

        def explode(*args, **kwargs):
            raise urllib.error.URLError("down")

        monkeypatch.setattr(notify.urllib.request, "urlopen", explode)

        Heartbeat("https://hc.example/uuid").ping()  # must not raise
