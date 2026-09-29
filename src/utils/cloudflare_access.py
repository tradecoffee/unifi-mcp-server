"""Optional Cloudflare Access JWT validation at the origin.

When the server sits behind Cloudflare Access (e.g. a Render deployment
fronted by a Zero Trust application), Cloudflare injects a signed JWT into
the ``Cf-Access-Jwt-Assertion`` header of every request it lets through.
Verifying that JWT at the origin proves the request traversed Access — a
caller who bypasses the Access hostname (say, by hitting the origin's raw
URL directly) cannot mint one, so their requests are rejected with 403.

Validation is opt-in: it runs only when both ``CF_ACCESS_TEAM_DOMAIN`` and
``CF_ACCESS_AUD`` are configured, and it layers on top of the mandatory
``MCP_AUTH_TOKEN`` bearer authentication rather than replacing it. The JWT
signature is checked against the team domain's published JWKS
(``https://<team>/cdn-cgi/access/certs``), and the ``iss``, ``aud`` and
``exp`` claims are all required and verified.
"""

from __future__ import annotations

import logging
from typing import Any

import anyio
import jwt
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

CF_ACCESS_JWT_HEADER = b"cf-access-jwt-assertion"


class CloudflareAccessVerifier:
    """Verify Cloudflare Access JWTs against the team domain's JWKS.

    Args:
        issuer: Expected ``iss`` claim, e.g. ``https://team.cloudflareaccess.com``
        audiences: Accepted ``aud`` values (Access application AUD tags)
        jwks_client: Optional PyJWKClient override (tests inject a stub)
    """

    def __init__(
        self,
        issuer: str,
        audiences: list[str],
        jwks_client: jwt.PyJWKClient | None = None,
    ) -> None:
        """Create a verifier for ``issuer`` accepting ``audiences``."""
        self.issuer = issuer
        self.audiences = audiences
        self._jwks_client = jwks_client or jwt.PyJWKClient(
            f"{issuer}/cdn-cgi/access/certs", cache_keys=True
        )

    def verify(self, token: str) -> dict[str, Any]:
        """Decode and validate a ``Cf-Access-Jwt-Assertion`` token.

        Args:
            token: The raw JWT string from the request header

        Returns:
            The verified JWT claims

        Raises:
            jwt.PyJWTError: For an unverifiable signature or a
                missing/invalid ``exp``, ``iss``, or ``aud`` claim
        """
        signing_key = self._jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=self.audiences,
            issuer=self.issuer,
            options={"require": ["exp", "iss", "aud"]},
        )


class CloudflareAccessGuard:
    """Pure ASGI middleware requiring a valid Cloudflare Access JWT.

    Rejects every HTTP request whose ``Cf-Access-Jwt-Assertion`` header is
    missing or fails verification with ``403``. Non-HTTP scopes (lifespan,
    websockets) pass through untouched.
    """

    def __init__(self, app: ASGIApp, verifier: CloudflareAccessVerifier) -> None:
        """Wrap ``app``, gating requests through ``verifier``."""
        self.app = app
        self._verifier = verifier

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        token = ""
        for name, value in scope.get("headers", []):
            if name.lower() == CF_ACCESS_JWT_HEADER:
                token = value.decode().strip()
                break

        if not token:
            await self._reject(scope, receive, send)
            return

        try:
            # PyJWKClient fetches keys over blocking urllib I/O on a cache
            # miss, so run verification off the event loop.
            await anyio.to_thread.run_sync(self._verifier.verify, token)
        except jwt.PyJWTError as exc:
            # Never log the token itself.
            logger.warning("Cloudflare Access JWT rejected: %s", type(exc).__name__)
            await self._reject(scope, receive, send)
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {
                "error": "forbidden",
                "error_description": "Valid Cloudflare Access token required",
            },
            status_code=403,
        )
        await response(scope, receive, send)
