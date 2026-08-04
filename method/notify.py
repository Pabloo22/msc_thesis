"""Outbound notifications for long, unattended GPU runs.

A trajectory is hours of rented GPU time that nobody is watching. Two things
are worth paying to learn about: that a run *broke* (so the box can be stopped
rather than billed for a dead process) and how fast it is *going* (so a
family's total cost can be projected before it is spent).

Mail goes out through Resend's HTTP API over the standard library alone. That
is deliberate rather than incidental: the image tag a rental box is booked
against is a digest of ``poetry.lock`` (see ``docs/cloud_setup.md``), so adding
a package purely to send an email would mean rebuilding and pushing ~17 GB
before the next rental could start.

Nothing in this module is allowed to raise. A notifier that kills the run it is
reporting on is strictly worse than no notifier, so every failure here is
logged and swallowed.

Two channels, because one of them cannot cover the case that matters most:

:class:`Notifier`
    Email, sent *by* the box over the box's network. It carries the detail --
    the traceback, the timing table, the projected cost -- and it is the one
    that usually arrives. It cannot arrive when the network is what failed,
    when the OOM killer sends SIGKILL (no ``finally`` block runs), or when the
    instance is preempted out from under the process.
:class:`Heartbeat`
    Pings an external watchdog on a fixed schedule. The *absence* of pings is
    the alarm, so nothing on the box has to be alive to raise it. It carries no
    detail at all.

They are complementary, not alternatives: the email says what went wrong, and
the heartbeat is what still works when the box is gone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from contextlib import AbstractContextManager
from types import TracebackType

from method.utils import DOTENV_PATH, load_dotenv

logger = logging.getLogger(__name__)

RESEND_API_KEY_ENV = "MSC_RESEND_API_KEY"
NOTIFY_EMAIL_ENV = "MSC_NOTIFY_EMAIL"
NOTIFY_FROM_ENV = "MSC_NOTIFY_FROM"
NOTIFY_TAG_ENV = "MSC_NOTIFY_TAG"
HEARTBEAT_URL_ENV = "MSC_HEARTBEAT_URL"

#: Resend's shared sender. It needs no verified domain, but can only deliver to
#: the address the Resend account was registered with -- which is all a
#: notifier that only ever mails its owner requires. Override with
#: ``MSC_NOTIFY_FROM`` once a domain is verified.
DEFAULT_SENDER = "onboarding@resend.dev"

_RESEND_ENDPOINT = "https://api.resend.com/emails"

#: Short enough that a wedged connection cannot stall a run behind a
#: notification, long enough to survive an ordinary slow response.
_TIMEOUT_SECONDS = 15.0

#: A failure email is worth retrying -- the network being unreliable is one of
#: the things it exists to report. Bounded low, because the run is either
#: already dead (nothing gained by waiting) or still going (nothing should
#: block it).
_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2.0

#: Load-bearing, not cosmetic. Cloudflare fronts the Resend API and bans
#: urllib's default ``Python-urllib/3.12`` agent on sight: every request comes
#: back ``403`` with Cloudflare error 1010 ("browser signature banned"), which
#: names neither the header nor Cloudflare and reads exactly like a rejected
#: API key. Any agent string that is not urllib's default gets through.
_USER_AGENT = "msc-thesis-notifier/1.0"


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> None:
    """POST ``payload`` as JSON, raising on any non-2xx response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            **headers,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        response.read()


class Notifier:
    """Email delivery, or a silent no-op when the environment is not set up.

    Unconfigured is the normal case for a laptop, a mock run and every test, so
    it is a state of this class rather than an error: callers never branch on
    whether notifications exist. :meth:`describe` is what a run logs at startup
    so that "no email arrived" can be told apart from "no email was ever going
    to be sent".
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        recipients: Sequence[str],
        sender: str = DEFAULT_SENDER,
        tag: str = "",
    ) -> None:
        self.api_key = api_key
        self.recipients = list(recipients)
        self.sender = sender
        #: Free-text marker for the subject line, e.g. the box's hostname. Two
        #: rentals working the same family otherwise send indistinguishable
        #: mail.
        self.tag = tag

    @classmethod
    def from_env(cls) -> Notifier:
        """Build from ``MSC_RESEND_API_KEY`` / ``MSC_NOTIFY_EMAIL``.

        Both are required; either one alone yields a disabled notifier, since
        a key with nobody to mail and an address with no way to reach it are
        equally unusable.
        """
        recipients = [
            address.strip()
            for address in os.environ.get(NOTIFY_EMAIL_ENV, "").split(",")
            if address.strip()
        ]
        return cls(
            api_key=os.environ.get(RESEND_API_KEY_ENV) or None,
            recipients=recipients,
            sender=os.environ.get(NOTIFY_FROM_ENV) or DEFAULT_SENDER,
            tag=os.environ.get(NOTIFY_TAG_ENV, ""),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.recipients)

    def describe(self) -> str:
        """One line for the startup log, never including the key itself."""
        if not self.enabled:
            missing = []
            if not self.api_key:
                missing.append(RESEND_API_KEY_ENV)
            if not self.recipients:
                missing.append(NOTIFY_EMAIL_ENV)
            return f"email notifications off (unset: {', '.join(missing)})"
        return f"email notifications on -> {', '.join(self.recipients)}"

    def send(self, subject: str, body: str) -> bool:
        """Mail ``body``, returning whether it was accepted.

        Never raises: the return value is for tests and for the heartbeat to
        decide whether the email channel is carrying anything, not something a
        run is expected to handle.
        """
        if not self.enabled:
            logger.debug("notifications disabled; dropping %r", subject)
            return False

        payload = {
            "from": self.sender,
            "to": self.recipients,
            "subject": f"[{self.tag}] {subject}" if self.tag else subject,
            "text": body,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                _post_json(_RESEND_ENDPOINT, payload, headers)
                logger.info("sent notification: %s", subject)
                return True
            except Exception as exc:  # noqa: BLE001 -- delivery is best-effort
                detail = _describe_http_error(exc)
                if attempt == _ATTEMPTS:
                    logger.warning(
                        "could not send notification %r: %s", subject, detail
                    )
                    return False
                logger.warning(
                    "notification attempt %d/%d failed (%s); retrying",
                    attempt,
                    _ATTEMPTS,
                    detail,
                )
                time.sleep(_RETRY_DELAY_SECONDS * attempt)
        return False


def _describe_http_error(exc: Exception) -> str:
    """Include an HTTP error's body, where the reason for a 4xx actually is.

    Resend answers a rejected sender or an unverified recipient with a 403 and
    a JSON explanation; without the body the log says only "HTTP Error 403",
    which is the least useful half of the message.
    """
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001 -- the body is a bonus, not required
            body = ""
        return f"HTTP {exc.code} {body}" if body else f"HTTP {exc.code}"
    return f"{type(exc).__name__}: {exc}"


class Heartbeat(AbstractContextManager["Heartbeat"]):
    """Fixed-period pings to an external watchdog (healthchecks.io or alike).

    The watchdog raises the alarm when pings stop for longer than a grace
    period, so something has to decide what "too long" means. Pinging once per
    finished checkpoint would make that a property of the workload -- a
    behaviour eval on a 7B model and one on the 0.5B proxy differ by more than
    an order of magnitude -- so the grace would need re-tuning per experiment,
    and would either cry wolf or stay quiet for hours. Pinging from a thread on
    a fixed period makes the expected interval a constant chosen here, so the
    grace is a small constant too and never needs tuning.

    The thread is a daemon: it must never be the reason a finished run fails to
    exit, and its pings mean "this process still exists", which stops being
    true at exactly the moment the process does.

    On the way out it pings ``/fail`` for an exception and ``/`` for success,
    so a failure that *does* have a working network raises the alarm at once
    rather than after the grace period. The timeout remains the backstop for
    the failures that cannot report themselves.
    """

    #: Chosen so the watchdog's grace can be a small constant. Frequent enough
    #: that a five-minute grace is many missed pings rather than one, cheap
    #: enough that it is invisible next to a GPU rental.
    PERIOD_SECONDS = 60.0

    def __init__(self, url: str | None, *, period: float = PERIOD_SECONDS) -> None:
        self.url = (url or "").rstrip("/") or None
        self.period = period
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls) -> Heartbeat:
        return cls(os.environ.get(HEARTBEAT_URL_ENV))

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def describe(self) -> str:
        if not self.enabled:
            return f"heartbeat off ({HEARTBEAT_URL_ENV} unset)"
        return f"heartbeat on, every {self.period:.0f}s"

    def ping(self, suffix: str = "") -> None:
        """Fire one ping. Silent on failure -- a missed ping *is* the signal."""
        if self.url is None:
            return
        try:
            # Same explicit agent as the mail path: healthchecks.io does not
            # filter on it today, but a watchdog that silently stops being
            # pinged is the one failure this class cannot report.
            request = urllib.request.Request(
                self.url + suffix, headers={"User-Agent": _USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                response.read()
        except Exception as exc:  # noqa: BLE001 -- best-effort by design
            logger.debug("heartbeat ping%s failed: %s", suffix, exc)

    def _loop(self) -> None:
        # wait() rather than sleep() so a finished run stops pinging
        # immediately instead of up to a full period later.
        while not self._stop.wait(self.period):
            self.ping()

    def __enter__(self) -> Heartbeat:
        if not self.enabled:
            return self
        self.ping("/start")
        self._thread = threading.Thread(
            target=self._loop, name="heartbeat", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_TIMEOUT_SECONDS)
        self.ping("/fail" if exc_type is not None else "")


def main() -> None:
    """Send a test notification, to check the wiring before spending GPU time.

    Everything here is best-effort at run time -- a notifier that cannot reach
    the network logs a warning and lets the trajectory carry on -- which is the
    right behaviour during a run and the wrong one for a setup check, where a
    silent failure means finding out only once something has already broken
    unnoticed. So this reports the outcome and exits non-zero, in the spirit of
    ``scripts/box_setup.sh``.

    Kept out of ``box_setup.sh`` itself because that script is meant to be
    re-run freely, and each invocation of this one costs an email.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heartbeat-only",
        action="store_true",
        help="ping the watchdog but send no mail",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(DOTENV_PATH)

    notifier = Notifier.from_env()
    heartbeat = Heartbeat.from_env()
    print(notifier.describe())
    print(heartbeat.describe())

    problems = []

    if heartbeat.enabled:
        # /start then /fail then a success ping: this exercises all three
        # endpoints and leaves the check in a passing state, so a test cannot
        # leave a watchdog that will alarm an hour later.
        heartbeat.ping("/start")
        heartbeat.ping("/fail")
        heartbeat.ping()
        print("pinged the watchdog (start, fail, then success -- check its log)")
    else:
        problems.append(
            f"{HEARTBEAT_URL_ENV} is unset: a box that is killed "
            "outright or loses its network cannot report itself"
        )

    if args.heartbeat_only:
        pass
    elif not notifier.enabled:
        problems.append(
            f"set {RESEND_API_KEY_ENV} and {NOTIFY_EMAIL_ENV} in {DOTENV_PATH}"
        )
    elif notifier.send(
        "test notification",
        "If this arrived, failure and progress emails will reach you.\n\n"
        f"Sender    {notifier.sender}\n"
        f"Recipients{' ' * 6}{', '.join(notifier.recipients)}\n\n"
        "Sent by `python -m method.notify`.\n",
    ):
        print(f"sent a test email to {', '.join(notifier.recipients)}")
    else:
        problems.append("the send failed; the warning above says why")

    for problem in problems:
        print(f"  problem: {problem}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
