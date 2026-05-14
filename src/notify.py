import os
import urllib.request

from src.log import log


def notify_phone(message: str, title: str = "ARC run") -> None:
    """Send a push notification to your phone via ntfy.sh.

    Set NTFY_TOPIC to the topic you subscribed to in the ntfy app.
    No-op if NTFY_TOPIC is unset.
    """
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return

    req = urllib.request.Request(
        f"{"https://ntfy.sh"}/{topic}",
        data=message.encode("utf-8"),
        headers={"Title": title},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log.warning(
            "Failed to send ntfy notification",
            error_type=type(e).__name__,
            error_message=str(e),
        )
