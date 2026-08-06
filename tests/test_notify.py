"""Tests for the failure/progress notification channels.

The property that matters most here is negative: a notifier exists to report
that something else broke, so it must not itself be able to break a run. Every
test that exercises a failure path asserts that the failure was swallowed.
"""

from __future__ import annotations

import json
import time
import urllib.error

import pytest

from method import notify
from method.notify import Heartbeat, Notifier, Throttle


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
def no_notify_env(monkeypatch, tmp_path):
    """Start every test from an unconfigured environment.

    Otherwise a developer's own .env (already loaded in some other test) would
    decide whether these pass, and a real email would be a possible outcome of
    running the suite.

    The send-rate state is redirected for the same reason in the other
    direction: the suite must not write to the repo root, and must not be able
    to leave behind a file that silences a real run's next hour of mail.
    """
    for var in (
        notify.RESEND_API_KEY_ENV,
        notify.NOTIFY_EMAIL_ENV,
        notify.NOTIFY_FROM_ENV,
        notify.NOTIFY_TAG_ENV,
        notify.NOTIFY_INTERVAL_ENV,
        notify.HEARTBEAT_URL_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(notify, "THROTTLE_STATE_PATH", tmp_path / "notify-state.json")


@pytest.fixture
def throttle(tmp_path):
    """An hourly throttle over a disposable state file."""
    return Throttle(tmp_path / "state.json", interval=3600.0)


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


class TestThrottle:
    """The provider's daily quota is the resource being protected here.

    Reaching it does not degrade gracefully -- every later mail is rejected,
    including the failure mail that is the reason any of this exists -- so the
    tests are about which mail gets dropped, not merely how many.
    """

    def test_first_claim_is_allowed_and_the_next_is_not(self, throttle):
        assert throttle.claim("exp3:ok").allowed
        throttle.record_sent("exp3:ok")

        assert not throttle.claim("exp3:ok").allowed

    def test_claiming_does_not_start_the_window_on_its_own(self, throttle):
        """A send that never reached the provider must not cost the window."""
        assert throttle.claim("exp3:ok").allowed

        assert throttle.claim("exp3:ok").allowed

    def test_window_reopens_once_the_interval_has_passed(self, throttle):
        throttle.record_sent("exp3:ok")
        _age_state(throttle, "exp3:ok", seconds=3601)

        assert throttle.claim("exp3:ok").allowed

    def test_keys_are_independent(self, throttle):
        """A family completing normally must not hide the first death."""
        throttle.record_sent("exp3:ok")

        assert not throttle.claim("exp3:ok").allowed
        assert throttle.claim("exp3:failed").allowed

    def test_a_sent_mail_says_how_many_it_stands_for(self, throttle):
        throttle.record_sent("exp3:ok")
        throttle.claim("exp3:ok")
        throttle.claim("exp3:ok")
        _age_state(throttle, "exp3:ok", seconds=3601)

        permit = throttle.claim("exp3:ok")

        assert permit.suppressed == 2
        assert "2 earlier report(s) were not mailed" in permit.annotate("body")

    def test_the_count_resets_once_one_gets_through(self, throttle):
        throttle.record_sent("exp3:ok")
        throttle.claim("exp3:ok")
        _age_state(throttle, "exp3:ok", seconds=3601)
        throttle.record_sent("exp3:ok")
        _age_state(throttle, "exp3:ok", seconds=3601)

        assert throttle.claim("exp3:ok").suppressed == 0

    def test_an_unkeyed_send_is_never_limited(self, throttle):
        throttle.record_sent("")

        assert throttle.claim("").allowed
        assert not throttle.path.exists()

    def test_a_corrupt_state_file_does_not_suppress_mail(self, throttle):
        """Erring towards sending: the alternative is silence for an hour."""
        throttle.path.write_text("{ this is not json")

        assert throttle.claim("exp3:failed").allowed

    def test_zero_interval_disables_the_limit(self, tmp_path):
        unlimited = Throttle(tmp_path / "state.json", interval=0.0)
        unlimited.record_sent("exp3:ok")

        assert unlimited.claim("exp3:ok").allowed
        assert not (tmp_path / "state.json").exists()

    def test_from_env_reads_minutes_and_falls_back_when_unparseable(self, monkeypatch):
        monkeypatch.setenv(notify.NOTIFY_INTERVAL_ENV, "15")
        assert Throttle.from_env().interval == 15 * 60

        monkeypatch.setenv(notify.NOTIFY_INTERVAL_ENV, "soon")
        assert Throttle.from_env().interval == (
            notify.DEFAULT_MIN_INTERVAL_MINUTES * 60
        )

    def test_notifier_from_env_is_throttled_by_default(self, monkeypatch):
        """The default has to be the safe one: a box is set up once, from a
        checklist, and the quota is reached weeks later."""
        monkeypatch.setenv(notify.RESEND_API_KEY_ENV, "re_test")
        monkeypatch.setenv(notify.NOTIFY_EMAIL_ENV, "me@example.com")

        assert Notifier.from_env().throttle.enabled

    def test_a_hand_built_notifier_is_not_throttled(self):
        """Nothing constructed in a test or a notebook should silently drop
        mail, nor write state into the repo root."""
        assert not Notifier(api_key="k", recipients=["me@example.com"]).throttle.enabled


class TestThrottledSending:
    def test_a_throttled_send_reports_not_sent_and_touches_no_network(
        self, monkeypatch, tmp_path
    ):
        calls = []
        monkeypatch.setattr(
            notify, "_post_json", lambda url, payload, headers: calls.append(payload)
        )
        notifier = Notifier(
            api_key="k",
            recipients=["me@example.com"],
            throttle=Throttle(tmp_path / "state.json", interval=3600.0),
        )

        assert notifier.send("first", "body", throttle_key="exp3:ok") is True
        assert notifier.send("second", "body", throttle_key="exp3:ok") is False
        assert len(calls) == 1

    def test_a_failed_send_leaves_the_window_open(self, monkeypatch, tmp_path):
        """Otherwise an outage would cost the next hour's reporting too."""
        monkeypatch.setattr(notify, "_post_json", _explode)
        monkeypatch.setattr(notify.time, "sleep", lambda _seconds: None)
        notifier = Notifier(
            api_key="k",
            recipients=["me@example.com"],
            throttle=Throttle(tmp_path / "state.json", interval=3600.0),
        )

        assert notifier.send("first", "body", throttle_key="exp3:ok") is False

        monkeypatch.setattr(notify, "_post_json", lambda url, payload, headers: None)
        assert notifier.send("second", "body", throttle_key="exp3:ok") is True

    def test_the_suppression_note_reaches_the_body(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(
            notify, "_post_json", lambda url, payload, headers: captured.update(payload)
        )
        throttle = Throttle(tmp_path / "state.json", interval=3600.0)
        notifier = Notifier(
            api_key="k", recipients=["me@example.com"], throttle=throttle
        )

        notifier.send("first", "first body", throttle_key="exp3:ok")
        notifier.send("dropped", "dropped body", throttle_key="exp3:ok")
        _age_state(throttle, "exp3:ok", seconds=3601)
        notifier.send("third", "third body", throttle_key="exp3:ok")

        assert captured["subject"] == "third"
        assert captured["text"].startswith("third body")
        assert "1 earlier report(s) were not mailed" in captured["text"]


def _explode(url, payload, headers):
    raise urllib.error.URLError("network is down")


def _age_state(throttle: Throttle, key: str, *, seconds: float) -> None:
    """Backdate ``key``'s last send, standing in for the clock moving on.

    Rewriting the state file rather than patching the clock keeps the interval
    honest: this is exactly what the next process to run would read.
    """
    state = json.loads(throttle.path.read_text())
    state[key]["last_sent"] = time.time() - seconds
    throttle.path.write_text(json.dumps(state))


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
