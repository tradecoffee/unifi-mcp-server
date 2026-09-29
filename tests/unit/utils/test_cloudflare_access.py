"""Unit tests for the optional Cloudflare Access JWT middleware."""

import time
from typing import Any
from unittest.mock import MagicMock

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.utils.cloudflare_access import CloudflareAccessGuard, CloudflareAccessVerifier

ISSUER = "https://team.cloudflareaccess.com"
AUD = "aud-tag-1"


def _rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _public_pem(private_key) -> Any:
    return private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _private_pem(private_key) -> bytes:
    return private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _token(private_key, *, aud=AUD, iss=ISSUER, exp=None, algorithm="RS256") -> str:
    claims = {
        "aud": aud,
        "iss": iss,
        "exp": exp if exp is not None else int(time.time()) + 300,
        "sub": "user@example.com",
    }
    return jwt.encode(claims, _private_pem(private_key), algorithm=algorithm, headers={"kid": "k1"})


def _verifier(public_key, *, audiences=None, jwks_client=None) -> CloudflareAccessVerifier:
    if jwks_client is None:
        jwks_client = MagicMock(spec=jwt.PyJWKClient)
        signing_key = MagicMock()
        signing_key.key = public_key
        jwks_client.get_signing_key_from_jwt.return_value = signing_key
    return CloudflareAccessVerifier(
        issuer=ISSUER, audiences=audiences or [AUD], jwks_client=jwks_client
    )


def make_client(verifier: CloudflareAccessVerifier) -> tuple[TestClient, list[str]]:
    """Build a tiny app wrapped in the guard; records whether it was reached."""
    reached: list[str] = []

    async def endpoint(request: Request) -> JSONResponse:
        reached.append(request.url.path)
        return JSONResponse({"ok": True})

    app = Starlette(
        routes=[Route("/mcp", endpoint, methods=["POST", "GET"])],
        middleware=[Middleware(CloudflareAccessGuard, verifier=verifier)],
    )
    return TestClient(app), reached


KEY = _rsa_key()
OTHER_KEY = _rsa_key()


def test_valid_token_reaches_app() -> None:
    client, reached = make_client(_verifier(_public_pem(KEY)))
    resp = client.post("/mcp", headers={"Cf-Access-Jwt-Assertion": _token(KEY)})
    assert resp.status_code == 200
    assert reached == ["/mcp"]


def test_missing_header_is_403() -> None:
    client, reached = make_client(_verifier(_public_pem(KEY)))
    resp = client.post("/mcp")
    assert resp.status_code == 403
    assert resp.json()["error"] == "forbidden"
    assert reached == []


def test_wrong_audience_is_403() -> None:
    client, reached = make_client(_verifier(_public_pem(KEY)))
    resp = client.post("/mcp", headers={"Cf-Access-Jwt-Assertion": _token(KEY, aud="someone-else")})
    assert resp.status_code == 403
    assert reached == []


def test_wrong_issuer_is_403() -> None:
    client, reached = make_client(_verifier(_public_pem(KEY)))
    resp = client.post(
        "/mcp",
        headers={"Cf-Access-Jwt-Assertion": _token(KEY, iss="https://evil.cloudflareaccess.com")},
    )
    assert resp.status_code == 403
    assert reached == []


def test_expired_token_is_403() -> None:
    client, reached = make_client(_verifier(_public_pem(KEY)))
    resp = client.post(
        "/mcp",
        headers={"Cf-Access-Jwt-Assertion": _token(KEY, exp=int(time.time()) - 60)},
    )
    assert resp.status_code == 403
    assert reached == []


def test_token_signed_by_other_key_is_403() -> None:
    client, reached = make_client(_verifier(_public_pem(KEY)))
    resp = client.post("/mcp", headers={"Cf-Access-Jwt-Assertion": _token(OTHER_KEY)})
    assert resp.status_code == 403
    assert reached == []


def test_hs256_token_is_403() -> None:
    client, reached = make_client(_verifier(_public_pem(KEY)))
    claims = {"aud": AUD, "iss": ISSUER, "exp": int(time.time()) + 300}
    token = jwt.encode(claims, "secret", algorithm="HS256")
    resp = client.post("/mcp", headers={"Cf-Access-Jwt-Assertion": token})
    assert resp.status_code == 403
    assert reached == []


def test_second_configured_audience_is_accepted() -> None:
    client, reached = make_client(_verifier(_public_pem(KEY), audiences=["aud-other", AUD]))
    resp = client.post("/mcp", headers={"Cf-Access-Jwt-Assertion": _token(KEY)})
    assert resp.status_code == 200
    assert reached == ["/mcp"]


def test_jwks_fetch_failure_is_403() -> None:
    jwks_client = MagicMock(spec=jwt.PyJWKClient)
    jwks_client.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientConnectionError("boom")
    client, reached = make_client(_verifier(None, jwks_client=jwks_client))
    resp = client.post("/mcp", headers={"Cf-Access-Jwt-Assertion": _token(KEY)})
    assert resp.status_code == 403
    assert reached == []


def test_verifier_uses_default_jwks_client() -> None:
    verifier = CloudflareAccessVerifier(issuer=ISSUER, audiences=[AUD])
    assert isinstance(verifier._jwks_client, jwt.PyJWKClient)
    assert verifier._jwks_client.uri == f"{ISSUER}/cdn-cgi/access/certs"
